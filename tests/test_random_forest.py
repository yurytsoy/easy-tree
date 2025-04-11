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
        train_df = self.df.filter(train_flag)
        train_y_true = self.y_true.filter(train_flag)
        val_df = self.df.filter(~train_flag)
        val_y_true = self.y_true.filter(~train_flag)

        with self.subTest("fit"):
            rf = RandomForest(n_estimators=10)
            rf.fit(train_df, y_true=train_y_true)

        with self.subTest("predict"):
            train_y_pred = rf.predict(train_df)
            train_acc = ((train_y_pred > train_y_true.mean()) == train_y_true.cast(bool)).mean()

            val_y_pred = rf.predict(val_df)
            val_acc = ((val_y_pred > val_y_true.mean()) == val_y_true.cast(bool)).mean()
            print(f"{train_acc:.4f} / {val_acc:.4f}")
            self.assertGreater(train_acc, 0.8191)
            self.assertGreater(val_acc, 0.8023)

        with self.subTest("feature importances"):
            expected = {
                'Sex': 0.5882921259383291,
                'Pclass': 0.13257927767445077,
                'Cabin': 0.09230503723215279,
                'Age': 0.07589993902851921,
                'Fare': 0.06468117362074698,
                'PassengerId': 0.031480656904779754,
                'Embarked': 0.014761789601021442,
            }
            self.assertDictEqual(rf.feature_importances_, expected)