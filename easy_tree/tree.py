from __future__ import annotations

from collections import defaultdict
import numpy as np
import orjson
import polars as pl

from .logic import ExpressionBuilder
from .entities import Node, TargetStats, BaseModel
from .usecases import find_split_cat, find_split_num, read_data, get_col


class DecisionTree(BaseModel):
    """
    Implementation of the CART regression tree with few tweaks for robustness.

    The tweaks include:
    * Considering only balanced splits for numerical variables.
    * Ignoring non-informative categories.
    * Handling missing values in numerical and categorical variables.
    * Handling variables with constant values.

    The convention: right condition is always precise and does not include missing values,
        unless it has missing value explicitly on the RHS.
    """

    def __init__(self, max_depth: int = 4, min_leaf_size: int = 10, max_features: int | float | str | None = None) -> None:
        self.max_depth = max(max_depth, 2)  # ignore values < 2
        self.min_leaf_size = max(min_leaf_size, 1)  # ignore values < 1
        self.max_features = max_features
        self.root_ = None
        self.feature_importances_ = None
        self.prediction_type_ = None

    @property
    def leaves(self) -> list[Node]:
        if not self.root_:  # if untrained tree.
            return []

        return [node for node in self.get_nodes() if node.left is None and node.right is None]

    def get_nodes(self, start: Node | None = None) -> list[Node]:
        """
        Get tree nodes starting from a given node. If no starting node is provided,
            then `root_` node is used as starting node.

        Returns
        -------
        list[Node]
            Flat list of tree nodes.
        """

        start = start or self.root_
        if not start:
            return []

        left = self.get_nodes(start.left) if start.left else []
        right = self.get_nodes(start.right) if start.right else []
        return [start] + left + right

    def validate_settings(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series):
        # min_leaf_size.
        if self.min_leaf_size >= len(y_true):
            raise RuntimeError("min_leaf_size must be smaller than data size")

        # max_features.
        if isinstance(self.max_features, int):
            if self.max_features > len(data.columns) or self.max_features < 1:
                raise RuntimeError("Integer `max_features` should be in the range [1, number_of_columns]")

        if isinstance(self.max_features, float):
            if self.max_features > 1.0 or self.max_features <= 0.0:
                raise RuntimeError("Fractional `max_features` should be in the range (0, 1.0].")

        if isinstance(self.max_features, str):
            if self.max_features not in ["sqrt", "log2"]:
                raise RuntimeError("String `max_features` should be one of: ['sqrt, 'log2']")

    def fit(self, data: pl.LazyFrame | pl.DataFrame | str, y_true: pl.Series | str) -> DecisionTree:
        """
        Args
        ----
        data : pl.LazyFrame | pl.DataFrame
        y_true : pl.Series
        """

        def get_max_features() -> int:
            if self.max_features is None:
                return len(data.columns)
            if isinstance(self.max_features, int):
                return self.max_features
            if isinstance(self.max_features, float):
                return max(1, int(self.max_features * len(data.columns)))
            if self.max_features == "sqrt":
                return round(np.sqrt(len(data.columns)).item())
            if self.max_features == "log2":
                return round(np.log2(len(data.columns)).item())
            raise NotImplementedError(f"Unsupported max_features value ({self.max_features})")

        # prepare data and target
        if isinstance(data, str) and isinstance(y_true, str):
            data = read_data(data)
            y_true = get_col(data, y_true)

        if y_true.name in data.columns:
            data = data.drop(y_true.name)

        self.validate_settings(data=data, y_true=y_true)

        # training.
        self.prediction_type_ = pl.Float64 if y_true.dtype.is_numeric() else pl.String
        self.root_ = self._make_node(node=Node(depth=1), data=data, y_true=y_true, max_features=get_max_features())

        # postprocessing: compute `feature_importances_`
        self._compute_feature_importance()
        return self

    def _compute_feature_importance(self):
        self.feature_importances_ = defaultdict(lambda: 0)  # 0 -- is a default value
        for node in self.get_nodes():
            if node.condition is None:
                continue
            # when node is split the split stats are written to the *parent's* `target_stats`
            self.feature_importances_[node.condition.column] += node.size * node.target_stats.score_reduction

        total_score = sum(self.feature_importances_.values())
        self.feature_importances_ = {
            colname: score / total_score
            for colname, score
            in sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True)
        }

    def _make_node(self, node: Node, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, max_features: int) -> Node | None:
        """
        Finalizes tree node, provided on input.

        Invariants:
        * any non-terminal node has defined split condition.
        * any node has information about target statistics.
        * either both `left` and `right` child nodes are defined or both are `None`
        * `right` child corresponds to the samples, where `condition = True` and `left` child to all remaining samples.
        """

        def maybe_get_in_memory_df() -> pl.DataFrame:
            """
            As we go further down the tree, the input data for each node becomes smaller and smaller.
            At some point it becomes small enough to safely use in-memory representation instead of the LazyFrame.
            """

            if isinstance(data, pl.DataFrame):
                return data

            MAX_IN_MEMORY_SIZE = 20_000  # convert to in-memory data frame  if the number of samples is below threshold.
            return data.collect() if len(y_true) <= MAX_IN_MEMORY_SIZE else data

        def init_node_stats():
            node.size = len(y_true)
            node.target_stats = TargetStats()
            if y_true.dtype.is_numeric():
                node.target_stats.mean = y_true.mean()
                node.target_stats.var = y_true.var()
            else:
                distr = {cat: count for cat, count in y_true.value_counts().rows()}
                node.target_stats.distr = {label: distr[label] for label in sorted(distr)}

        init_node_stats()

        if node.depth > self.max_depth:
            return None  # max depth reached.

        if len(y_true) < self.min_leaf_size:
            return None  # insufficient data.

        data = maybe_get_in_memory_df()

        # find the best split
        best_split = None
        columns = np.random.choice(data.columns, max_features, replace=False)\
            if max_features < len(data.columns)\
            else data.columns
        columns_to_drop = []
        for colname in columns:
            if data.schema[colname].is_numeric():
                cur_split = find_split_num(data=data, colname=colname, y_true=y_true)
            else:
                cur_split = find_split_cat(data=data, colname=colname, y_true=y_true)
            if cur_split is None or cur_split.best_split_eval is None:
                columns_to_drop.append(colname)
                continue

            if best_split is None or best_split.best_split_eval < cur_split.best_split_eval:
                best_split = cur_split

        if best_split is None or best_split.best_split_eval <= 0:
            return node

        if columns_to_drop:
            if max_features == len(data.columns):
                max_features = max(1, max_features-len(columns_to_drop))  # update max_features only if it is same as the number of columns
            data = data.drop(columns_to_drop)
            max_features = min(max_features, len(data.columns))  # in order to avoid cases, when max_features > number of columns

        # note: split score is written to the *parent* node!
        node.target_stats.score_reduction = best_split.best_split_eval

        # make child nodes
        right_condition = best_split.best_split_condition
        right_mask = right_condition.apply(data)
        left_mask = ~right_mask
        node.right = self._make_node(
            node=Node(depth=node.depth+1, parent=node),
            data=data.filter(right_mask),
            y_true=y_true.filter(right_mask),
            max_features=max_features,
        )
        node.left = self._make_node(
            node=Node(depth=node.depth+1, parent=node),
            data=data.filter(left_mask),
            y_true=y_true.filter(left_mask),
            max_features = max_features,
        )

        # either both left and right nodes are present, or both are absent. Otherwise, the prediction is not possible.
        node.left = node.left if node.right is not None else None
        node.right = node.right if node.left is not None else None

        # node's condition is defined only if it is a non-terminal node.
        if node.right is None and node.left is None:
            node.condition = None
        else:
            node.condition = right_condition

        return node

    def predict(self, data: pl.LazyFrame) -> pl.Series:
        res = None
        for leaf in self.leaves:
            cur_flag = leaf.full_condition.apply(data)

            if res is None:
                res = pl.Series(values=[None] * len(cur_flag), dtype=self.prediction_type_)

            if self.prediction_type_.is_numeric():
                res = res.set(cur_flag, leaf.target_stats.mean)
            else:
                max_class, max_count = max(leaf.target_stats.distr.items(), key=lambda item: item[1])
                res = res.set(cur_flag, max_class)

        if res.is_null().any():
            # In some cases data can contain samples, which are not falling into any of the existing leafs.
            #   For that case use `root_` as a failover.
            empty_preds = res.is_null()
            if self.prediction_type_.is_numeric():
                res = res.set(empty_preds, self.root_.target_stats.mean)
            else:
                max_class, max_count = max(self.root_.target_stats.distr.items(), key=lambda item: item[1])
                res = res.set(empty_preds, max_class)

        return res

    def serialize(self) -> dict:
        return {
            "root_": self.root_.serialize() if self.root_ is not None else None,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "feature_importances_": self.feature_importances_,
            "prediction_type_": str(self.prediction_type_) if self.prediction_type_ else None,
        }

    @staticmethod
    def deserialize(data: dict) -> DecisionTree:
        res = DecisionTree(
            max_depth=data["max_depth"],
            min_leaf_size=data["min_leaf_size"],
        )
        res.root_ = Node.deserialize(data["root_"]) if data["root_"] is not None else None
        res.feature_importances_ = data["feature_importances_"]

        pred_types = {"Float64": pl.Float64, "String": pl.String, None: None}
        res.prediction_type_ = pred_types[data["prediction_type_"]]
        return res
