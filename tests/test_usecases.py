import unittest

import numpy as np
import polars as pl

from easy_tree.usecases import find_split_cat, find_split_num, sample


class TestUsecases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = pl.scan_csv("tests/data/titanic_train.csv")
        cls.y_true = cls.df.select("Survived").collect().to_series()

    def test_find_split_num_no_missing_values(self):
        res = find_split_num(self.df, colname="Fare", y_true=self.y_true)
        self.assertEqual(len(res.split_pts), 3)
        self.assertIsNotNone(res.best_idx)
        for split_eval in res.split_evals:
            self.assertLessEqual(split_eval, res.best_split_eval)

    def test_find_split_num_with_missing_values(self):
        res = find_split_num(self.df, colname="Age", y_true=self.y_true)
        self.assertIsNotNone(res.best_idx)
        self.assertTrue(np.nan in res.split_pts)
        for split_eval in res.split_evals:
            self.assertLessEqual(split_eval, res.best_split_eval)

    def test_find_split_num_constant(self):
        df = self.df.with_columns(
            pl.Series(
                values=np.ones(len(self.y_true))
            ).alias("all_ones")
        ).collect().lazy()
        res = find_split_num(df, colname="all_ones", y_true=self.y_true)
        self.assertIsNone(res.best_idx)
        self.assertIsNone(res.best_split_point)
        self.assertIsNone(res.best_split_eval)
        self.assertIsNone(res.best_split_condition)

    def test_find_split_num_skewed_percentile(self):
        """
        The 50 and 75 percentiles are equal to 3, meaning that the right split is empty. Such percentiles should be discarded.
        """
        res = find_split_num(self.df, colname="Pclass", y_true=self.y_true)
        self.assertEqual(res.best_idx, 0)
        self.assertEqual(res.split_pts, [2.0])
        self.assertEqual(str(res.best_split_condition), "(Pclass > 2.0) AND (Pclass != None)")

    def test_find_split_num_all_missing(self):
        with self.subTest("NaN"):
            df = self.df.with_columns(
                pl.Series(
                    values=[np.nan] * len(self.y_true)
                ).alias("all_missing")
            ).collect().lazy()
            res = find_split_num(df, colname="all_missing", y_true=self.y_true)
            self.assertIsNone(res.best_idx)
            self.assertIsNone(res.best_split_point)
            self.assertIsNone(res.best_split_eval)
            self.assertIsNone(res.best_split_condition)

        with self.subTest("None"):
            with self.assertRaises(TypeError):
                find_split_num(self.df, colname="Name", y_true=self.y_true)

    def test_find_split_num_not_numeric(self):
        with self.assertRaises(TypeError):
            find_split_num(self.df, colname="Name", y_true=self.y_true)

    def test_find_split_cat_no_missing_values(self):
        res = find_split_cat(self.df, colname="Sex", y_true=self.y_true)
        for split_eval in res.split_evals:
            self.assertLessEqual(split_eval, res.best_split_eval)

    def test_find_split_cat_too_many_categories(self):
        res = find_split_cat(self.df, colname="Name", y_true=self.y_true)
        self.assertIsNone(res.best_idx)
        self.assertIsNone(res.best_split_point)
        self.assertIsNone(res.best_split_eval)
        self.assertIsNone(res.best_split_condition)

    def test_find_split_cat_with_missing_values(self):
        res = find_split_cat(self.df, colname="Cabin", y_true=self.y_true)
        self.assertEqual(res.split_pts, [None])
        self.assertEqual(res.best_idx, 0)
        self.assertEqual(res.best_split_point, None)
        self.assertEqual(str(res.best_split_condition), "Cabin == None")

    def test_find_split_cat_constant(self):
        df = self.df.with_columns(
            pl.Series(
                values=["foo"] * len(self.y_true)
            ).alias("constant")
        ).collect().lazy()
        res = find_split_cat(df, colname="constant", y_true=self.y_true)
        self.assertIsNone(res.best_idx)
        self.assertIsNone(res.best_split_point)
        self.assertIsNone(res.best_split_eval)
        self.assertIsNone(res.best_split_condition)

    def test_sample(self):
        with self.subTest("With index"):
            res = sample(self.df, add_index=True)
            idx_col = res.select("__index__").collect().to_series()
            self.assertLess(idx_col.n_unique(), len(idx_col) * 2 / 3)  # according to the bootstrap probability (https://stats.stackexchange.com/questions/173520/random-forests-out-of-bag-sample-size).

        with self.subTest("Without index"):
            res = sample(self.df, add_index=False)
            self.assertFalse("__index__" in res.columns)
            id_col = res.select("PassengerId").collect().to_series()
            self.assertLess(id_col.n_unique(), len(idx_col) * 2 / 3)  # according to the bootstrap probability (https://stats.stackexchange.com/questions/173520/random-forests-out-of-bag-sample-size).
