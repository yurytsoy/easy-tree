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

    def assertLeavesSufficientAndDoNotOverlap(self, tree: et.DecisionTree, data: pl.DataFrame | pl.LazyFrame):
        """
        Checks that leaf paths do not overlap and tree produces exactly one prediction for each data sample.
        """
        data = data.collect() if isinstance(data, pl.LazyFrame) else data
        mask = np.array([0] * len(data))
        for leaf in tree.leaves:
            mask = mask + leaf.full_condition.apply(data).cast(int).to_numpy()
        self.assertEqual(np.nansum(mask), len(data))
        self.assertEqual(np.nanmin(mask), 1)  # every data sample is accounted for
        self.assertEqual(np.nanmax(mask), 1)  # ... exactly one time
        self.assertFalse(np.isnan(mask).any())  # nans are taken care of

    def compute_accuracy(
        self,
        tree: et.DecisionTree,
        train_data: pl.DataFrame | pl.LazyFrame,
        train_y_true: pl.Series,
        val_data: pl.DataFrame | pl.LazyFrame,
        val_y_true: pl.Series,
    ) -> tuple[float, float]:
        thr = train_y_true.mean()
        pred_train = (tree.predict(train_data) > thr).cast(float)
        accuracy_train = (pred_train == train_y_true).mean()
        pred_val = (tree.predict(val_data) > thr).cast(float)
        accuracy_val = (pred_val == val_y_true).mean()
        return accuracy_train, accuracy_val

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
            self.assertLeavesSufficientAndDoNotOverlap(tree, self.df)

        with self.subTest("predict"):
            pred = tree.predict(self.df)
            self.assertFalse(pred.is_null().any())

        with self.subTest("evaluate"):
            # compute accuracy
            accuracy_train, accuracy_val = self.compute_accuracy(
                tree=tree,
                train_data=train_df, train_y_true=train_y_true,
                val_data=val_df, val_y_true=val_y_true,
            )
            print(f"{accuracy_train:.4f} / {accuracy_val:.4f}")

            self.assertGreater(accuracy_train, 0.817)
            self.assertGreater(accuracy_val, 0.779)

        with self.subTest("feature importance"):
            expected = {
                'Sex': 0.6679081775805736,
                'Pclass': 0.13750220987739162,
                'Fare': 0.07150023687560125,
                'Cabin': 0.05887837713232438,
                'PassengerId': 0.030081550266882327,
                'SibSp': 0.020981015889506866,
                'Parch': 0.008684832763910761,
                'Embarked': 0.0044635996138091385,
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
            self.assertLeavesSufficientAndDoNotOverlap(tree, self.df)

    def test_fit_from_filename(self):
        tree = et.DecisionTree(max_depth=6)

        with self.subTest("fit"):
            np.random.seed(42)
            tree.fit(
                data="tests/data/titanic_train.csv", y_true="Survived",
            )
            self.assertLeavesSufficientAndDoNotOverlap(tree, self.df)

        with self.subTest("importance"):
            expected = {
                'Sex': 0.6445801824625548,
                'Pclass': 0.1605773859463015,
                'Cabin': 0.07097931642521112,
                'Embarked': 0.041554119567714064,
                'Fare': 0.04099654751628022,
                'PassengerId': 0.01195148817993547,
                'SibSp': 0.019732239166551025,
                'Parch': 0.009628720735451818,
            }
            self.assertImportanceEqual(tree, expected)

        with self.subTest("Accuracy"):
            pred_thr = self.y_true.mean()
            pred_train = (tree.predict(self.df) > pred_thr).cast(float)
            accuracy_train = (pred_train == self.y_true).mean()
            self.assertGreater(accuracy_train, 0.8069)

    def test_fit_max_features(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        train_df = self.df.filter(train_flag)
        train_y_true = self.y_true.filter(train_flag)
        val_df = self.df.filter(~train_flag)
        val_y_true = self.y_true.filter(~train_flag)

        ress = dict()
        for cur_value in [None, 5, 0.6, "sqrt", "log2"]:
            np.random.seed(42)
            tree = et.DecisionTree(max_features=cur_value).fit(train_df, train_y_true)
            train_acc, val_acc = self.compute_accuracy(
                tree=tree,
                train_data=train_df, train_y_true=train_y_true,
                val_data=val_df, val_y_true=val_y_true,
            )
            ress[cur_value] = train_acc, val_acc
            self.assertGreater(val_acc, 0.68)
        # print(ress)

    def test_fit_max_features_wrong_settings(self):
        with self.subTest("Bad integer"):
            for cur_value in [20, 0, -1]:
                with self.assertRaises(RuntimeError):
                    et.DecisionTree(max_features=cur_value).fit(self.df, self.y_true)

        with self.subTest("Bad float"):
            for cur_value in [1.01, 0.0, -0.1]:
                with self.assertRaises(RuntimeError):
                    et.DecisionTree(max_features=cur_value).fit(self.df, self.y_true)

        with self.subTest("Bad string"):
            for cur_value in ["foo", "Sqrt", "log"]:
                with self.assertRaises(RuntimeError):
                    et.DecisionTree(max_features=cur_value).fit(self.df, self.y_true)

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
            tree = et.DecisionTree()
            tree.fit(data=train_df, y_true=train_y_true)
            self.assertLeavesSufficientAndDoNotOverlap(tree, self.df)
            self.assertTrue(all([leaf.target_stats.distr is not None for leaf in tree.leaves]))

        with self.subTest("Predict"):
            pred_train = tree.predict(train_df)
            self.assertFalse(pred_train.is_null().any())
            accuracy_train = (pred_train == train_y_true).mean()

            pred_val = tree.predict(data.filter(~train_flag))
            self.assertFalse(pred_val.is_null().any())
            accuracy_val = (pred_val == y_true.filter(~train_flag)).mean()
            print(f"{accuracy_train:.4f} / {accuracy_val:.4f}")

            self.assertGreater(accuracy_train, 0.8136)
            self.assertGreater(accuracy_val, 0.7906)

        with self.subTest("Save/Load"):
            filename = "/tmp/tree.json"
            tree.save(filename)
            tree_2 = et.DecisionTree.load(filename)
            pred_val_2 = tree_2.predict(data.filter(~train_flag))
            self.assertTrue(all(pred_val == pred_val_2))
