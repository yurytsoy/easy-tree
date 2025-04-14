from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, min_leaf_size: int = 10, max_features: int | float | str | None = 1.0, n_jobs: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees_ = []
        self.feature_importances_ = None
        self.oob_counts_ = None
        self._seeds = []  # array of RNG seeds to reproducibility of sampling

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
        self._seeds = np.random.randint(10000, size=self.n_estimators)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_trees = [executor.submit(self._fit_tree, data, y_true, k) for k in range(self.n_estimators)]
            for future in as_completed(future_to_trees):
                try:
                    res = future.result()
                    self.trees_.append(res.tree)
                    self.oob_counts_[res.oob_idxs] += 1
                except Exception as e:
                    print(e)

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

    def _fit_tree(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, tree_idx: int=None) -> FitTreeReport:
        seed = self._seeds[tree_idx] if tree_idx is not None else None
        cur_data = sample(data, add_index=True, seed=seed)
        idxs = get_col(cur_data, "__index__")
        cur_y_true = y_true[idxs]

        tree = DecisionTree(
            max_depth=self.max_depth, min_leaf_size=self.min_leaf_size, max_features=self.max_features
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
