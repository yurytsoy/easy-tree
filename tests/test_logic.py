import numpy as np
import polars as pl
import unittest

from easy_tree.logic import Operator, AtomicExpression, AndExpression, OrExpression, NotExpression, ExpressionBuilder


class TestAtomicExpression(unittest.TestCase):
    def test_apply(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        expr = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)
        res = expr.apply(df)
        self.assertTrue((df.filter(res).select("Age").collect().to_series() > 30).all())

    def test_apply_numpy(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        expr = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)
        res = expr.apply_numpy(df.select("Age").collect().to_series().to_numpy())
        self.assertTrue((df.filter(res).select("Age").collect().to_series() > 30).all())

    def test_not_(self):
        with self.subTest(">"):
            item = AtomicExpression(colname="Age", operator=Operator.greater, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.less_or_equal, rhs=10))

        with self.subTest(">="):
            item = AtomicExpression(colname="Age", operator=Operator.greater_or_equal, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.less, rhs=10))

        with self.subTest("<"):
            item = AtomicExpression(colname="Age", operator=Operator.less, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.greater_or_equal, rhs=10))

        with self.subTest("<="):
            item = AtomicExpression(colname="Age", operator=Operator.less_or_equal, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.greater, rhs=10))

        with self.subTest("=="):
            item = AtomicExpression(colname="Age", operator=Operator.equal, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.not_equal, rhs=10))

        with self.subTest("!="):
            item = AtomicExpression(colname="Age", operator=Operator.not_equal, rhs=10)
            self.assertEqual(item.not_(), AtomicExpression(colname="Age", operator=Operator.equal, rhs=10))


class TestAndExpression(unittest.TestCase):
    def test_apply(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)
        right = AtomicExpression(colname="Pclass", operator=Operator.equal, rhs=2)
        expr = AndExpression(left, right)
        res = expr.apply(df)
        res_df = df.filter(res).collect()
        self.assertTrue((res_df.select("Age").to_series() > 30).all())
        self.assertTrue((res_df.select("Pclass").to_series() == 2).all())

    def test_apply_numpy(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Age", operator=Operator.greater, rhs=20)
        right = AtomicExpression(colname="Age", operator=Operator.less_or_equal, rhs=60)
        expr = AndExpression(left, right)
        age_vals = df.select("Age").collect().to_series().to_numpy()
        res = expr.apply_numpy(age_vals)
        res_age_vals = age_vals[res]
        self.assertTrue(((20 < res_age_vals) & (res_age_vals <= 60)).all())

    def test_apply_numpy_cat(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="Q")
        right = AtomicExpression(colname="Embarked", operator=Operator.not_equal, rhs=None)
        expr = AndExpression(left, right)
        embarked_vals = df.select("Embarked").collect().to_series().to_numpy()
        res = expr.apply_numpy(embarked_vals)
        res_embarked_vals = embarked_vals[res]
        self.assertTrue(((res_embarked_vals == "Q") & (res_embarked_vals is not None)).all())

    def test_apply_nested(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)
        right = AtomicExpression(colname="Pclass", operator=Operator.equal, rhs=2)
        expr = AndExpression(left, right)

        left2 = AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="Q")
        expr2 = AndExpression(left2, expr)
        res = expr2.apply(df)
        res_df = df.filter(res).collect()
        self.assertTrue((res_df.select("Age").to_series() > 30).all())
        self.assertTrue((res_df.select("Pclass").to_series() == 2).all())
        self.assertTrue((res_df.select("Embarked").to_series() == "Q").all())


class TestOrExpression(unittest.TestCase):
    def test_apply(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)
        right = AtomicExpression(colname="Pclass", operator=Operator.equal, rhs=2)
        expr = OrExpression(left, right)
        res = expr.apply(df)
        res_df = df.filter(res).collect()
        res_series = (res_df.select("Age").to_series() > 30) | (res_df.select("Pclass").to_series() == 2)
        self.assertTrue(res_series.all())

    def test_apply_numpy(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Age", operator=Operator.less, rhs=20)
        right = AtomicExpression(colname="Age", operator=Operator.greater_or_equal, rhs=60)
        expr = AndExpression(left, right)
        age_vals = df.select("Age").collect().to_series().to_numpy()
        res = expr.apply_numpy(age_vals)
        res_age_vals = age_vals[res]
        self.assertTrue(((20 > res_age_vals) & (res_age_vals >= 60)).all())

    def test_apply_numpy_cat(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        left = AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="Q")
        right = AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="C")
        expr = AndExpression(left, right)
        embarked_vals = df.select("Embarked").collect().to_series().to_numpy()
        res = expr.apply_numpy(embarked_vals)
        res_embarked_vals = embarked_vals[res]
        self.assertTrue(((res_embarked_vals == "Q") | (res_embarked_vals == "C")).all())


class TestNotExpression(unittest.TestCase):
    def test_apply(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        item = AtomicExpression(colname="Age", operator=Operator.equal, rhs=None)
        expr = NotExpression(item)
        res = expr.apply(df)
        res_series = df.filter(res).collect().select("Age").to_series()
        self.assertEqual(res_series.is_null().sum() + res_series.is_nan().sum(), 0)

    def test_apply_numpy(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        item = AtomicExpression(colname="Age", operator=Operator.equal, rhs=None)
        expr = NotExpression(item)
        age_vals = df.select("Age").collect().to_series().to_numpy()
        res = expr.apply_numpy(age_vals)
        res_age_vals = age_vals[res]
        self.assertEqual(np.isnan(res_age_vals).sum(), 0)

    def test_not_(self):
        item = AtomicExpression(colname="Age", operator=Operator.equal, rhs=None)
        expr = NotExpression(item)
        not_expr = expr.not_()
        self.assertEqual(not_expr, item)


class TestExpressionBuilder(unittest.TestCase):
    def test_apply(self):
        df = pl.scan_csv("tests/data/titanic_train.csv")
        start = AtomicExpression(colname="Age", operator=Operator.not_equal, rhs=None)
        expr = (ExpressionBuilder(start)
                .and_(
                    OrExpression(
                        left=AtomicExpression(colname="Pclass", operator=Operator.equal, rhs=1),
                        right=AtomicExpression(colname="Pclass", operator=Operator.equal, rhs=2),
                    )
                ))
        res_mask = expr.apply(df)
        res_data = df.filter(res_mask).collect()
        res_age = res_data.select("Age").to_series()
        self.assertEqual(res_age.is_null().sum() + res_age.is_nan().sum(), 0)
        res_pclass = res_data.select("Pclass").to_series()
        self.assertTrue(res_pclass.is_in([1, 2]).all())

    def test_not_(self):
        with self.subTest("AtomicExpression"):
            item = AtomicExpression(colname="Age", operator=Operator.greater, rhs=50)
            res = ExpressionBuilder(item).not_()
            self.assertEqual(res.current, AtomicExpression(colname="Age", operator=Operator.greater, rhs=50).not_())

        with self.subTest("NotExpression"):
            item = AtomicExpression(colname="Age", operator=Operator.equal, rhs=50)
            res = ExpressionBuilder(item).not_()
            self.assertEqual(res.current, AtomicExpression(colname="Age", operator=Operator.equal, rhs=50).not_())

        with (self.subTest("AndExpression")):
            item = AndExpression(
                left=AtomicExpression(colname="Age", operator=Operator.greater, rhs=50),
                right=AtomicExpression(colname="Fare", operator=Operator.less, rhs=20),
            )
            res = ExpressionBuilder(item).not_()
            self.assertEqual(res.current.serialize(), NotExpression(item).serialize())

        with self.subTest("OrExpression"):
            item = OrExpression(
                left=AtomicExpression(colname="Age", operator=Operator.greater, rhs=50),
                right=AtomicExpression(colname="Fare", operator=Operator.less, rhs=20),
            )
            res = ExpressionBuilder(item).not_()
            self.assertEqual(res.current.serialize(), NotExpression(item).serialize())
