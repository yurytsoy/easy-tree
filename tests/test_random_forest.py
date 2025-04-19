import unittest

import numpy as np
import polars as pl

from easy_tree import RandomForest


class TestRandomForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = pl.scan_csv("tests/data/titanic_train.csv").drop("Name")
        cls.y_true = cls.df.select("Survived").collect().to_series()

    def test_fit_regression(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        train_df = self.df.filter(train_flag).collect()
        train_y_true = self.y_true.filter(train_flag)
        val_df = self.df.filter(~train_flag).collect()
        val_y_true = self.y_true.filter(~train_flag)

        for _ in range(10):
            with self.subTest("fit"):
                np.random.seed(42)
                rf = RandomForest(n_estimators=50)
                rf.fit(train_df, y_true=train_y_true)

            with self.subTest("predict"):
                train_y_pred = rf.predict(train_df)
                pred_thr = train_y_true.mean()
                train_acc = ((train_y_pred > pred_thr) == train_y_true.cast(bool)).mean()

                val_y_pred = rf.predict(val_df)
                val_acc = ((val_y_pred > pred_thr) == val_y_true.cast(bool)).mean()
                print(f"{train_acc:.4f} / {val_acc:.4f}")
                # self.assertGreater(train_acc, 0.819)
                # self.assertGreater(val_acc, 0.819)

    def test_fit_classification(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        y_true = self.y_true.cast(str)
        train_df = self.df.filter(train_flag).collect()
        train_y_true = y_true.filter(train_flag)
        val_df = self.df.filter(~train_flag).collect()
        val_y_true = y_true.filter(~train_flag)

        ress = []
        for _ in range(10):
            with self.subTest("fit"):
                np.random.seed(42)
                rf = RandomForest(n_estimators=30)
                rf.fit(train_df, y_true=train_y_true)
                ress.append(rf.serialize())

            with self.subTest("predict"):
                train_y_pred = rf.predict(train_df)
                train_acc = (train_y_pred == train_y_true).mean()

                val_y_pred = rf.predict(val_df)
                val_acc = (val_y_pred == val_y_true).mean()
                print(f"{train_acc:.4f} / {val_acc:.4f}")
                # self.assertGreater(train_acc, 0.819)
                # self.assertGreater(val_acc, 0.796)
        # TODO: compare random forests

    def test_serialization_deserialization_regression(self):
        with self.subTest("fit"):
            np.random.seed(42)
            rf = RandomForest(n_estimators=10)
            rf.fit(self.df, y_true=self.y_true)
            pred = rf.predict(self.df)

        with self.subTest("serialize"):
            rf_data = rf.serialize()
            self.assertIsNotNone(rf_data)

        with self.subTest("deserialize"):
            rf2 = RandomForest.deserialize(rf_data)
            pred2 = rf2.predict(self.df)
            self.assertTrue((pred == pred2).all())

    def test_serialization_deserialization_classification(self):
        with self.subTest("fit"):
            np.random.seed(42)
            rf = RandomForest(n_estimators=10)
            rf.fit(self.df, y_true=self.y_true.cast(str))
            pred = rf.predict(self.df)
            self.assertEqual(pred.dtype, pl.String)

        with self.subTest("serialize"):
            rf_data = rf.serialize()
            self.assertIsNotNone(rf_data)

        with self.subTest("deserialize"):
            rf2 = RandomForest.deserialize(rf_data)
            pred2 = rf2.predict(self.df)
            self.assertTrue((pred == pred2).all())
