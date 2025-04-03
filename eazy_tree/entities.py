from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import polars as pl

from .logic import BaseExpression, ExpressionBuilder, AtomicExpression


@dataclass
class TargetStats:
    score_reduction: float | None = None  # reduction of a score *after* the split at the node.
    mean: float | None = None  # [regression] average value of the target, corresponding to the node.
    var: float | None = None  # [regression] variance of the target, corresponding to the node.
    distr: dict[str, int] | None = None  # [classification] target distribution.


@dataclass
class Node:
    """
    Node for a binary tree.
    """

    depth: int
    condition: BaseExpression | None = None  # the most *recent* conditions that lead to the node.
    right: Node | None = None  # "Yes" branch: corresponds to the `greater` for numerical or `equal` for categorical variables.
    left: Node | None = None  # "No" branch: corresponds to the `less or equal` for numerical or `not equal` for categorical variables.
    parent: Node | None = None
    size: int | None = None  # number of samples from the training data that fall into the node.
    target_stats: TargetStats | None = None

    def __repr__(self) -> str:
        return f"d{self.depth} | {self.size} | {self.target_stats.mean:.6f} | {self.condition}"

    @property
    def full_condition(self) -> BaseExpression:
        """
        Returns all conditions starting from the very top parent node and down to the current node.

        The conditions are sorted by depth.
        """
        if self.parent is None:  # if root node
            return self.condition

        is_left = id(self.parent.left) == id(self)  # checks whether node is located in the left branch, and should negate the full path.
        parent_cond = self.parent.full_condition  # "path" to the current node
        if is_left:  # negate the last condition ("path" segment)
            if isinstance(parent_cond, AtomicExpression):
                parent_cond = ExpressionBuilder(parent_cond).not_().current
            else:  # if not atomic then it is an AND expression
                if self.depth == 2:  # if parent node is a root, negate the full parent condition
                    parent_cond = ExpressionBuilder(parent_cond).not_().current
                else:  # if parent node is not root, then each terms in the AND expression are compound expressions
                    parent_cond.right = ExpressionBuilder(parent_cond.right).not_().current

        if self.condition is not None:  # if non-root non-terminal node
            return ExpressionBuilder(parent_cond).and_(self.condition).current

        return parent_cond

    def serialize(self) -> dict:
        return {
            "depth": self.depth,
            "condition": self.condition.serialize() if self.condition is not None else None,
            "right": self.right.serialize() if self.right is not None else None,
            "left": self.left.serialize() if self.left is not None else None,
            "parent": None,  # always None in order to avoid hitting the recursion limit.
            "size": self.size,
            "target_stats": self.target_stats,
        }

    @staticmethod
    def deserialize(data: dict) -> Node:
        res = Node(
            depth=data["depth"],
            condition=BaseExpression.deserialize(data["condition"]),
            size=data["size"],
            target_stats=TargetStats(**data["target_stats"]) if data["target_stats"] is not None else None,
            left=Node.deserialize(data["left"]) if data["left"] is not None else None,
            right=Node.deserialize(data["right"]) if data["right"] is not None else None,
        )
        if res.left:
            res.left.parent = res
        if res.right:
            res.right.parent = res
        return res


@dataclass
class SplitReport:
    split_pts: list[float | int | str] = field(default_factory=list)
    split_evals: list[float] = field(default_factory=list)
    best_split_condition: BaseExpression | None = None  # split condition, which is True for the *Yes* branch.
    best_idx: int | None = None

    @property
    def colname(self) -> str | None:
        if not self.best_split_condition:
            return None
        return self.best_split_condition.column

    @property
    def best_split_point(self) -> float | int | str | None:
        if self.best_idx is None:
            return None

        return self.split_pts[self.best_idx]

    @property
    def best_split_eval(self) -> float | None:
        if self.best_idx is None:
            return None

        return self.split_evals[self.best_idx]


class BaseSplitScoring:
    data: pl.LazyFrame | pl.DataFrame
    y_true: pl.Series
    column: str
    split_conditions: list[BaseExpression]
    split_scores: list[float]
    split_points: list[int | float | str | None]

    def __init__(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, column: str):
        self.data = data
        self.y_true = y_true
        self.column = column
        self.split_conditions = []
        self.split_scores = []
        self.split_points = []

    def add_split_condition(self, condition: BaseExpression, split_point: int | float | str | None):
        """
        Add condition for splitting and compute split score.
        """
        pass

    def get_report(self) -> SplitReport:
        best_idx = int(np.argmax(self.split_scores)) if len(self.split_scores) > 0 else None
        return SplitReport(
            split_pts=self.split_points,
            split_evals=self.split_scores,
            best_idx=best_idx,
            best_split_condition=self.split_conditions[best_idx] if best_idx is not None else None,
        )


class VarianceScoring(BaseSplitScoring):
    def __init__(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, column: str):
        super().__init__(data, y_true, column=column)
        self.variance = self.y_true.var()

    def add_split_condition(self, condition: BaseExpression, split_point: int | float | str | None):
        true_mask = condition.apply(self.data)
        if true_mask.all() or true_mask.sum() <= 1:
            return  # one of the splits is empty or contains only 1 element.
        false_mask = ~true_mask
        if false_mask.all() or false_mask.sum() <= 1:
            return  # one of the splits is empty or contains only 1 element.

        gt_true = self.y_true.filter(true_mask)
        gt_false = self.y_true.filter(false_mask)
        num_false = len(gt_false)
        num_true = len(self.y_true) - num_false

        false_var = gt_false.var() or 0
        true_var = gt_true.var() or 0
        split_var = (false_var * num_false + true_var * num_true) / len(self.y_true)
        reduction = self.variance - split_var
        self.split_conditions.append(condition)
        self.split_scores.append(reduction)
        self.split_points.append(split_point)


class EntropyScoring(BaseSplitScoring):
    @staticmethod
    def _entropy(s: pl.Series) -> float:
        return s.value_counts().select("count").to_series().entropy()

    def __init__(self, data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, column: str):
        super().__init__(data, y_true, column=column)
        self.entropy = self._entropy(self.y_true)

    def add_split_condition(self, condition: BaseExpression, split_point: int | float | str | None):
        """
        Add condition for splitting and compute split score.
        """
        true_mask = condition.apply(self.data)
        if true_mask.all() or true_mask.sum() <= 1:
            return  # one of the splits is empty or contains only 1 element.
        false_mask = ~true_mask
        if false_mask.all() or false_mask.sum() <= 1:
            return  # one of the splits is empty or contains only 1 element.

        ent_true = self._entropy(self.y_true.filter(true_mask))
        ent_false = self._entropy(self.y_true.filter(false_mask))
        num_false = false_mask.sum()
        num_true = len(self.y_true) - num_false

        reduction = self.entropy - (num_true * ent_true + num_false * ent_false) / len(self.y_true)
        self.split_conditions.append(condition)
        self.split_scores.append(reduction)
        self.split_points.append(split_point)
