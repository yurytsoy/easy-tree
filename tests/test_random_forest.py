import unittest

import numpy as np
import polars as pl

from easy_tree import RandomForest


class TestRandomForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = pl.scan_csv("tests/data/titanic_train.csv")
        cls.y_true = cls.df.select("Survived").collect().to_series()

    def test_fit_regression(self):
        rng = np.random.default_rng(42)
        train_flag = pl.Series(values=rng.choice([True, False], size=len(self.y_true), p=[0.8, 0.2]))
        df = self.df.drop("Name")
        train_df = df.filter(train_flag).collect()
        train_y_true = self.y_true.filter(train_flag)
        val_df = df.filter(~train_flag).collect()
        val_y_true = self.y_true.filter(~train_flag)

        with self.subTest("fit"):
            np.random.seed(42)
            rf = RandomForest(n_estimators=500, max_features="log2")
            rf.fit(train_df, y_true=train_y_true)

        with self.subTest("predict"):
            train_y_pred = rf.predict(train_df)
            pred_thr = train_y_true.mean()
            train_acc = ((train_y_pred > pred_thr) == train_y_true.cast(bool)).mean()

            val_y_pred = rf.predict(val_df)
            val_acc = ((val_y_pred > pred_thr) == val_y_true.cast(bool)).mean()
            print(f"{train_acc:.4f} / {val_acc:.4f}")
            # self.assertGreater(train_acc, 0.817)
            # self.assertGreater(val_acc, 0.819)
