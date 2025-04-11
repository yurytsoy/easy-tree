from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np
import polars as pl

from .entities import BaseModel
from .tree import DecisionTree
from .usecases import read_data, get_col, sample


@dataclasses.dataclass
class FitTreeReport:
    tree: DecisionTree  # trained tree.
    oob_idxs: list[int]  # indices for the out of bag samples.


class RandomForest(BaseModel):
    n_estimators: int
    trees_: list[DecisionTree] | None

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, min_leaf_size: int = 10, n_jobs: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_jobs = n_jobs
        self.trees_ = []
        self.feature_importances_ = None
        self.oob_counts_ = None

    def fit(self, data: pl.LazyFrame | pl.DataFrame | str, y_true: pl.Series | str) -> BaseModel:
        # prepare data and target
        if isinstance(data, str) and isinstance(y_true, str):
            data = read_data(data)
            y_true = get_col(data, y_true)

        if y_true.name in data.columns:
            data = data.drop(y_true.name)

        self.trees_ = []

        # train trees
        self.oob_counts_ = np.zeros(len(y_true))
        for _ in range(self.n_estimators):
            res = self._fit_tree(data, y_true)
            self.trees_.append(res.tree)
            self.oob_counts_[res.oob_idxs] += 1

        # compute feature importance as average importance
        self.feature_importances_ = defaultdict(lambda: 0)
        for tree in self.trees_:
            for feature, importance in tree.feature_importances_.items():
                self.feature_importances_[feature] += importance
        total_imp = sum(self.feature_importances_.values())
        for feature in self.feature_importances_.keys():
            self.feature_importances_[feature] /= total_imp
        self.feature_importances_ = dict(self.feature_importances_)

        return self

    def _fit_tree(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series) -> FitTreeReport:
        cur_data = sample(data, add_index=True)
        idxs = get_col(cur_data, "__index__")
        cur_y_true = y_true[idxs]
        tree = DecisionTree(
            max_depth=self.max_depth, min_leaf_size=self.min_leaf_size
        ).fit(cur_data.drop("__index__"), cur_y_true)
        return FitTreeReport(tree, oob_idxs=sorted(set(range(len(y_true))) - set(idxs)))

    def predict(self, data: pl.LazyFrame) -> pl.Series:
        if not self.trees_:
            raise RuntimeError("The model is not trained.")

        pred_type = self.trees_[0].prediction_type_
        if pred_type.is_numeric():
            # regression => compute average prediction
            pred = None
            for tree in self.trees_:
                tree_pred = tree.predict(data)
                if pred is None:
                    pred = tree_pred
                    continue
                pred += tree_pred
            pred /= len(self.trees_)
            return pred

        else:
            # classification => collect distribution of classes
            ...
