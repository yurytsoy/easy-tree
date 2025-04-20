from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, min_leaf_size: int = 10, max_features: int | float | str | None = 1.0, n_jobs: int | None = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_features = max_features
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
        # Note: parallel execution with ThreadPoolExecutor and ProcessPoolExecutor does not provide much improvement and
        #   conversely yields longer runtime (the example is below) even though CPU load does increase (up to ~200-250%).
        #   The exact reason for the lack of speedup is unclear and would be good to address in the future.
        #   More broad testing against datasets of different sizes and different RF settings is required.
        self.oob_counts_ = np.zeros(len(y_true))
        seeds = np.random.randint(10000, size=self.n_estimators)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for k, seed in zip(range(self.n_estimators), seeds):
                futures.append(executor.submit(self._fit_tree, data.clone(), y_true.clone(), seed))
            for future in futures:
                res = future.result()
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

    def _fit_tree(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, seed: int=None) -> FitTreeReport:
        cur_data = sample(data, add_index=True, seed=seed)
        idxs = get_col(cur_data, "__index__")

        tree = DecisionTree(
            max_depth=self.max_depth, min_leaf_size=self.min_leaf_size, max_features=self.max_features
        ).fit(cur_data.drop("__index__"), y_true[idxs])
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
            class_counts = dict()
            for tree in self.trees_:
                tree_pred = tree.predict(data)
                for class_label in tree_pred.unique():
                    if class_label not in class_counts:
                        class_counts[class_label] = (tree_pred == class_label).cast(int)
                    else:
                        class_counts[class_label] = class_counts[class_label] + (tree_pred == class_label).cast(int)

            # The tie-breaking is unstable, so that class with maximal counts can be determined ...
            #   khm, indeterministically.
            # In order to fix that, the classes are sorted by their frequency, the most frequent class going first.
            def get_ordered_training_class_probabilities() -> dict[str, float]:
                res = defaultdict(lambda: 0)
                for tree in self.trees_:
                    for label, count in tree.root_.target_stats.distr.items():
                        res[label] += count
                total = sum(res.values())
                return {label: count / total for label, count in res.items()}

            probs = get_ordered_training_class_probabilities()
            classes = list(probs)
            pred = (pl.DataFrame([class_counts[label] for label in classes if label in class_counts])
                    .map_rows(lambda row: classes[np.argmax(row)])
                    .rename({"map": "prediction"})
                    .to_series())
            return pred

    def serialize(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "trees_": [tree.serialize() for tree in self.trees_],
            "feature_importances_": self.feature_importances_,
            "oob_counts_": self.oob_counts_,
        }

    @classmethod
    def deserialize(cls, data: dict) -> RandomForest:
        res = RandomForest(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            min_leaf_size=data["min_leaf_size"],
            max_features=data["max_features"],
            n_jobs=data["n_jobs"],
        )
        res.trees_ = [DecisionTree.deserialize(tree_info) for tree_info in data["trees_"]]
        res.feature_importances_ = data["feature_importances_"]
        res.oob_counts_ = data["oob_counts_"]
        return res
