from __future__ import annotations

import numpy as np
import polars as pl

from .logic import Operator, AtomicExpression, ExpressionBuilder
from .entities import SplitReport, BaseSplitScoring, VarianceScoring, EntropyScoring


def read_data(filename: str) -> pl.LazyFrame:
    if filename.endswith(".csv"):
        return pl.scan_csv(filename)
    if filename.endswith(".parquet") or filename.endswith(".pq") or filename.endswith(".pqt"):
        return pl.scan_parquet(filename)
    raise NotImplementedError(f"Unknown data file format: {filename}")


def get_col(data: pl.LazyFrame | pl.DataFrame, colname: str) -> pl.Series:
    if isinstance(data, pl.DataFrame):
        return data[colname]
    else:
        return data.select(colname).collect().to_series()


def get_scoring_method(data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, colname: str) -> BaseSplitScoring:
    if y_true.dtype.is_numeric():
        return VarianceScoring(data=data, y_true=y_true, column=colname)
    return EntropyScoring(data=data, y_true=y_true, column=colname)


def find_split_num(
    data: pl.LazyFrame | pl.DataFrame,
    colname: str,
    y_true: pl.Series,
    scoring: BaseSplitScoring | None = None
) -> SplitReport:
    """
    Heuristic method for finding a split point, that maximizes variance reduction after the split.

    The candidate points, which are checked:
    * 25, 50, 75 percentiles
    * np.nan -- if missing values are present
    """

    class ColStats:
        """
        Wrapper for the polars `describe` result.
        """

        def __init__(self, column: pl.Series):
            self.stats = column.describe()

        @property
        def is_dull(self) -> bool:
            is_all_missing = len(self.stats) == 2
            is_all_nan = not is_all_missing and np.isnan(self.std)
            is_constant = not is_all_missing and self.std == 0
            return is_all_missing or is_constant or is_all_nan

        @property
        def has_missing_values(self) -> bool:
            null_count_idx = 1  # index at which the statistics contain information about missing values.
            return self.stats["value"][null_count_idx] > 0

        @property
        def std(self) -> float:
            return self.stats["value"][3]

        def perc(self, perc_name: str) -> float:
            idx = self.stats["statistic"].index_of(perc_name)
            return self.stats["value"][idx]

    scoring = scoring or get_scoring_method(data=data, y_true=y_true, colname=colname)

    col = get_col(data, colname)
    if not col.dtype.is_numeric():
        raise TypeError(f"{colname} is not numeric or all-None.")

    col_stats = ColStats(col)
    if col_stats.is_dull:
        return scoring.get_report()

    for perc_name in ["25%", "50%", "75%"]:
        split_pt = col_stats.perc(perc_name)
        split_condition = ExpressionBuilder(
            AtomicExpression(colname=colname, operator=Operator.greater, rhs=split_pt)
        ).and_(AtomicExpression(colname=colname, operator=Operator.not_equal, rhs=None)).current
        scoring.add_split_condition(split_condition, split_point=split_pt)

    if col_stats.has_missing_values:
        condition = AtomicExpression(colname=colname, operator=Operator.not_equal, rhs=None)
        scoring.add_split_condition(condition, np.nan)

    return scoring.get_report()


def find_split_cat(
    data: pl.LazyFrame | pl.DataFrame,
    colname: str,
    y_true: pl.Series,
    min_count: int = 10,
    scoring: BaseSplitScoring | None = None
) -> SplitReport:
    """
    Method for finding a split point, that maximizes variance reduction after the split.

    Parameters
    ----------
    min_count : int, default=10
        Minimum required number of samples with a certain category.
        All categories below this requirement are discarded.

    Returns
    -------
    SplitReport
        Results of splitting.
    """

    scoring = scoring or get_scoring_method(data=data, y_true=y_true, colname=colname)

    col = get_col(data, colname)
    val_counts = col.value_counts(sort=True)
    if len(val_counts) == 1:    # either all missing or constant
        return scoring.get_report()

    for category, count in val_counts.rows():
        if count < min_count:
            continue

        # evaluate split point
        scoring.add_split_condition(
            AtomicExpression(colname=colname, operator=Operator.equal, rhs=category),
            split_point=category,
        )

    return scoring.get_report()
