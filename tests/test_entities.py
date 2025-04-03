import unittest

import polars as pl

import eazy_tree as et
from eazy_tree.logic import AtomicExpression, Operator, ExpressionBuilder


class TestNode(unittest.TestCase):
    def test_full_condition_depth_1_2(self):
        data = pl.LazyFrame({
            "foo": [None] * 10 + list(range(100)),
            "bar": list(range(100)) + [None] * 10
        })
        foo_cond = ExpressionBuilder(
            AtomicExpression(colname="foo", operator=Operator.greater, rhs=10)
        ).and_(AtomicExpression(colname="foo", operator=Operator.not_equal, rhs=None)).current

        with self.subTest("no parent, no child nodes"):
            node = et.Node(depth=1)
            node.condition = foo_cond
            self.assertEqual(str(node.full_condition), "(foo > 10) AND (foo != None)")
            res = node.full_condition.apply(data)
            self.assertEqual(res.is_null().sum(), 0)
            self.assertEqual(res.sum(), 89)

        with self.subTest("with parent, right child, no cond"):
            node1 = et.Node(depth=1)
            node1.condition = foo_cond
            node2 = et.Node(depth=2, parent=node1)
            node1.right = node2
            self.assertEqual(str(node2.full_condition), "(foo > 10) AND (foo != None)")
            res_right = node2.full_condition.apply(data)

        with self.subTest("with parent, left child, no cond"):
            node1 = et.Node(depth=1)
            node1.condition = foo_cond
            node2 = et.Node(depth=2, parent=node1)
            node1.left = node2
            self.assertEqual(str(node2.full_condition), "NOT ((foo > 10) AND (foo != None))")
            res_left = node2.full_condition.apply(data)

            # check that res_right and res_left cover all cases and do not overlap
            self.assertTrue((res_left | res_right).all())
            self.assertFalse((res_left & res_right).any())

    def test_full_condition_depth_3(self):
        data = pl.LazyFrame({
            "foo": [None] * 10 + list(range(100)),
            "bar": list(range(100)) + [None] * 10
        })
        foo_cond = ExpressionBuilder(
            AtomicExpression(colname="foo", operator=Operator.greater, rhs=10)
        ).and_(AtomicExpression(colname="foo", operator=Operator.not_equal, rhs=None)).current
        bar_cond = ExpressionBuilder(
            AtomicExpression(colname="bar", operator=Operator.greater, rhs=10)
        ).and_(AtomicExpression(colname="bar", operator=Operator.not_equal, rhs=None)).current

        node1 = et.Node(depth=1, condition=foo_cond)
        node2 = et.Node(depth=2, parent=node1, condition=bar_cond)
        node1.right = node2
        res_node2 = node2.full_condition.apply(data)

        with self.subTest("right child, full data"):
            node3_r = et.Node(depth=3, parent=node2)
            node2.right = node3_r
            self.assertEqual(str(node3_r.full_condition), "((foo > 10) AND (foo != None)) AND ((bar > 10) AND (bar != None))")
            res_r = node3_r.full_condition.apply(data)
            self.assertEqual(res_r.is_null().sum(), 0)

        with self.subTest("left child, full data"):
            node3_l = et.Node(depth=3, parent=node2)
            node2.left = node3_l
            self.assertEqual(str(node3_l.full_condition), "((foo > 10) AND (foo != None)) AND (NOT ((bar > 10) AND (bar != None)))")
            # check that node3_r is not affected and has the same full condition as before.
            self.assertEqual(str(node3_r.full_condition), "((foo > 10) AND (foo != None)) AND ((bar > 10) AND (bar != None))")
            res_l = node3_l.full_condition.apply(data)
            self.assertEqual(res_l.is_null().sum(), 0)
            self.assertEqual(res_node2.sum() + 10, (res_l | res_r).sum())  # because res_l includes missing values for `bar` and full data has those missing values!
            self.assertFalse((res_l & res_r).any())

        with self.subTest("data from node 2"):
            data_2 = data.filter(node1.full_condition.apply(data))
            res_r_2 = node3_r.full_condition.apply(data_2)
            res_l_2 = node3_l.full_condition.apply(data_2)
            self.assertEqual(res_r.is_null().sum(), 0)
            self.assertEqual(res_l.is_null().sum(), 0)
            self.assertEqual(res_r_2.sum() + res_l_2.sum(), data_2.collect().shape[0])
            self.assertFalse((res_l_2 & res_r_2).any())  # mutually exclusive ...
            self.assertTrue((res_l_2 ^ res_r_2).all())  # ... and populate all available positions.
