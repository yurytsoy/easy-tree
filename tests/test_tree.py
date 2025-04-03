import unittest

import numpy as np
import polars as pl

import easy_tree as et


class TestRegressionTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = pl.scan_csv("tests/data/titanic_train.csv")
        cls.y_true = cls.df.select("Survived").collect().to_series()

    def assertImportanceEqual(self, tree: et.DecisionTree, expected: dict):
        tree_imps = tree.feature_importances_
        for colname, importance in expected.items():
            self.assertAlmostEqual(importance, tree_imps[colname], places=15)

    def test_fit(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        train_df = self.df.filter(train_flag)
        train_y_true = self.y_true.filter(train_flag)
        val_df = self.df.filter(~train_flag)
        val_y_true = self.y_true.filter(~train_flag)

        with self.subTest("fit"):
            tree = et.DecisionTree(max_depth=6)
            tree.fit(
                data=train_df, y_true=train_y_true,
            )
            df_len = len(train_y_true)
            self.assertEqual(df_len, sum([leaf.size for leaf in tree.leaves]))

        with self.subTest("predict"):
            pred = tree.predict(self.df)
            self.assertFalse(pred.is_null().any())

        with self.subTest("evaluate"):
            # compute accuracy
            thr = self.y_true.mean()
            pred_train = (tree.predict(train_df) > thr).cast(float)
            accuracy_train = (pred_train == train_y_true).mean()
            pred_val = (tree.predict(val_df) > thr).cast(float)
            accuracy_val = (pred_val == val_y_true).mean()
            print(f"{accuracy_train:.4f} / {accuracy_val:.4f}")

            self.assertGreater(accuracy_train, 0.827)
            self.assertGreater(accuracy_val, 0.802)

        with self.subTest("feature importance"):
            expected = {
                'Sex': 0.6136906971387358,
                'Pclass': 0.12634046096492582,
                'SibSp': 0.08287575329334973,
                'Cabin': 0.05409892186022007,
                'Age': 0.04112586438724043,
                'Embarked': 0.03992830777911069,
                'Fare': 0.03337783138261036,
                'PassengerId': 0.004639260426636065,
                'Parch': 0.00392290276717106,
            }
            self.assertImportanceEqual(tree, expected)

    def test_fit_min_leaf_size(self):
        df = pl.scan_parquet("tests/data/titanic_train.pq")

        with self.subTest("min_leaf_size = 10"):
            tree = et.DecisionTree(min_leaf_size=1)
            tree.fit(
                data=df, y_true=self.y_true,
            )
            self.assertTrue(all(
                [node.size > tree.min_leaf_size for node in tree.get_nodes()]
            ))

        with self.subTest("min_leaf_size = 10"):
            tree1 = et.DecisionTree(min_leaf_size=10)
            tree1.fit(
                data=df, y_true=self.y_true,
            )
            self.assertTrue(all(
                [node.size > tree1.min_leaf_size for node in tree1.get_nodes()]
            ))

        with self.subTest("min_leaf_size = 100"):
            tree2 = et.DecisionTree(min_leaf_size=100)
            tree2.fit(
                data=df, y_true=self.y_true,
            )
            self.assertTrue(all(
                [node.size > tree2.min_leaf_size for node in tree2.get_nodes()]
            ))

        with self.subTest("min_leaf_size is too large"):
            tree3 = et.DecisionTree(min_leaf_size=1000)  # > data size
            with self.assertRaises(RuntimeError):
                tree3.fit(data=df, y_true=self.y_true)

    def test_fit_non_lazy(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        train_df = self.df.filter(train_flag)
        train_y_true = self.y_true.filter(train_flag)

        with self.subTest("fit"):
            tree = et.DecisionTree(max_depth=6)
            tree.fit(
                data=train_df.collect(), y_true=train_y_true,
            )
            df_len = len(train_y_true)
            self.assertEqual(df_len, sum([leaf.size for leaf in tree.leaves]))

        with self.subTest("Importance"):
            expected = {
                'Sex': 0.6136906971387358,
                'Pclass': 0.12634046096492582,
                'SibSp': 0.08287575329334973,
                'Cabin': 0.05409892186022007,
                'Age': 0.04112586438724043,
                'Embarked': 0.03992830777911069,
                'Fare': 0.03337783138261036,
                'PassengerId': 0.004639260426636065,
                'Parch': 0.00392290276717106,
            }
            self.assertImportanceEqual(tree, expected)

    def test_fit_from_filename(self):
        tree = et.DecisionTree(max_depth=6)

        with self.subTest("fit"):
            tree.fit(
                data="tests/data/titanic_train.csv", y_true="Survived",
            )
            df_len = len(pl.read_csv("tests/data/titanic_train.csv"))
            self.assertEqual(df_len, sum([leaf.size for leaf in tree.leaves]))

        with self.subTest("importance"):
            expected = {
                'Sex': 0.5919627751068731,
                'Pclass': 0.14746937245422082,
                'SibSp': 0.06981350649198743,
                'Cabin': 0.06518523881037531,
                'Age': 0.05093652498733797,
                'Embarked': 0.03178960639915491,
                'Fare': 0.029560853402885803,
                'Parch': 0.01124111027755316,
                'PassengerId': 0.0020410120696114048,
            }
            self.assertImportanceEqual(tree, expected)

        with self.subTest("Accuracy"):
            pred_thr = self.y_true.mean()
            pred_train = (tree.predict(self.df) > pred_thr).cast(float)
            accuracy_train = (pred_train == self.y_true).mean()
            self.assertGreater(accuracy_train, 0.824)

    def test_save_load(self):
        tree = et.DecisionTree()
        filename = "/tmp/tree.json"

        with self.subTest("Not-trained tree"):
            tree.save(filename)
            tree_2 = et.DecisionTree.load(filename)
            self.assertEqual(tree.root_, tree_2.root_)
            self.assertEqual(tree.max_depth, tree_2.max_depth)
            self.assertEqual(tree.min_leaf_size, tree_2.min_leaf_size)
            self.assertEqual(tree.feature_importances_, tree_2.feature_importances_)
            self.assertEqual(tree.prediction_type_, tree_2.prediction_type_)

        with self.subTest("Trained tree"):
            df = pl.scan_parquet("tests/data/titanic_train.pq")
            tree.fit(
                data=df, y_true=self.y_true,
            )
            pred = tree.predict(df)

            tree.save(filename)
            tree_2 = et.DecisionTree.load(filename)
            # self.assertEqual(tree.root_, tree_2.root_)  # hits max recursion depth due to `parent`
            self.assertEqual(tree.max_depth, tree_2.max_depth)
            self.assertEqual(tree.min_leaf_size, tree_2.min_leaf_size)
            self.assertEqual(tree.feature_importances_, tree_2.feature_importances_)
            self.assertEqual(tree.prediction_type_, tree_2.prediction_type_)

            # predictions should be the same
            pred_2 = tree_2.predict(df)
            np.testing.assert_array_equal(pred.to_numpy(), pred_2.to_numpy())

    def test_fit_classification(self):
        schema = self.df.schema
        schema["Survived"] = pl.String
        data = pl.read_csv("tests/data/titanic_train.csv", schema=schema)
        y_true = data.select("Survived").to_series()

        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(y_true), p=[0.8, 0.2]))
        train_df = data.filter(train_flag)
        train_y_true = y_true.filter(train_flag)

        with self.subTest("Fit"):
            tree = et.DecisionTree(max_depth=6)
            tree.fit(data=train_df, y_true=train_y_true)
            self.assertEqual(len(train_df), sum([leaf.size for leaf in tree.leaves]))
            self.assertTrue(all([leaf.target_stats.distr is not None for leaf in tree.leaves]))

        with self.subTest("Predict"):
            pred_train = tree.predict(train_df)
            self.assertFalse(pred_train.is_null().any())
            accuracy_train = (pred_train == train_y_true).mean()

            pred_val = tree.predict(data.filter(~train_flag))
            self.assertFalse(pred_val.is_null().any())
            accuracy_val = (pred_val == y_true.filter(~train_flag)).mean()
            print(f"{accuracy_train:.4f} / {accuracy_val:.4f}")

            self.assertGreater(accuracy_train, 0.844)
            self.assertGreater(accuracy_val, 0.802)

        with self.subTest("Save/Load"):
            filename = "/tmp/tree.json"
            tree.save(filename)
            tree_2 = et.DecisionTree.load(filename)
            pred_val_2 = tree_2.predict(data.filter(~train_flag))
            self.assertTrue(all(pred_val == pred_val_2))
