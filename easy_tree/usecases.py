from __future__ import annotations

import numpy as np
import os
import polars as pl
import string

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


def get_random_filename(dir: str, suffix: str) -> str:
    filename = "".join(np.random.choice(list(string.ascii_uppercase + string.ascii_lowercase), 20))
    return os.path.join(dir, f"{filename}{suffix}")


def sample(data: pl.LazyFrame | pl.DataFrame, add_index: bool = True, seed: int = 42) -> pl.LazyFrame | pl.DataFrame:
    """
    Resamples the provided data frame and returns a resulting frame that has the same dimensions as the input one.

    There is a discussion about `sample` implementation, however the result contains less number of samples:
    * https://github.com/pola-rs/polars/issues/3933

    Note:
    * If lazy data is used or the data size is large the result is saved in the temporary file in the `/tmp` folder.

    Args
    ----
    data: LazyFrame | DataFrame
    add_index: bool, default=True
        Indicates whether column containing sample indices from the original data will be added.
        The column name is `__index__`
    seed : int, default=42
        Seed for reproducibility of sampling.
    """

    idx_colname = "__index__"

    num_rows = data.select(pl.len()).collect().item() if isinstance(data, pl.LazyFrame) else len(data)
    if add_index:
        data = data.with_columns(pl.arange(num_rows).alias(idx_colname))

    if num_rows < 1000_000 or isinstance(data, pl.DataFrame):
        if isinstance(data, pl.LazyFrame):
            return data.collect().sample(n=num_rows, with_replacement=True, seed=seed).lazy()
        else:
            return data.sample(n=num_rows, with_replacement=True, seed=seed)

    else:
        def sample_batch(batch: pl.DataFrame) -> pl.DataFrame:
            return batch.sample(n=len(batch), with_replacement=True, seed=int(rng.integers(65536)))

        rng = np.random.default_rng(seed)
        tmp_file = get_random_filename(dir="/tmp", suffix=".pq")
        data.map_batches(sample_batch).sink_parquet(tmp_file)
        return read_data(tmp_file)


def get_scoring_method(data: pl.LazyFrame | pl.DataFrame, y_true: pl.Series, column: pl.Series) -> BaseSplitScoring:
    if y_true.dtype.is_numeric():
        return VarianceScoring(data=data, y_true=y_true, column=column)
    return EntropyScoring(data=data, y_true=y_true, column=column)


def find_split_num(
    data: pl.LazyFrame | pl.DataFrame,
    colname: str,
    y_true: pl.Series,
    scoring: BaseSplitScoring | None = None
) -> SplitReport | None:
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
            xs = column.to_numpy()
            self.has_missing_values = np.isnan(xs).any()
            if xs.any():
                percentiles = np.percentile(xs, [0, 25, 50, 75, 100])  # min, max + quartiles.
                self.percentiles = sorted(set(percentiles[1:4]))
                self.empty_std = percentiles[0] == percentiles[4]
            else:
                self.percentiles = None
                self.empty_std = True

        @property
        def is_dull(self) -> bool:
            return self.percentiles is None or self.empty_std

    col = get_col(data, colname)
    if not col.dtype.is_numeric():
        raise TypeError(f"{colname} is not numeric or all-None.")

    col_stats = ColStats(col)
    if col_stats.is_dull:
        return None

    scoring = scoring or get_scoring_method(data=data, y_true=y_true, column=col)
    for split_pt in col_stats.percentiles:
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
) -> SplitReport | None:
    """
    Method for finding a split point, that maximizes variance reduction after the split.

    Parameters
    ----------
    min_count : int, default=10
        Minimum required number of samples with a certain category.
        All categories below this requirement are discarded.

    Returns
    -------
    SplitReport | None
        Results of splitting.
    """

    col = get_col(data, colname)
    val_counts = col.value_counts(sort=True)
    if len(val_counts) == 1:    # either all missing or constant
        return None

    scoring = scoring or get_scoring_method(data=data, y_true=y_true, column=col)
    has_missing_values = col.is_null().sum() > 0
    for category, count in val_counts.rows():  # None is included!
        if count < min_count:
            continue

        # evaluate split point
        if has_missing_values and category is not None:
            split_condition = ExpressionBuilder(
                AtomicExpression(colname=colname, operator=Operator.equal, rhs=category)
            ).and_(AtomicExpression(colname=colname, operator=Operator.not_equal, rhs=None)).current
        else:
            split_condition = AtomicExpression(colname=colname, operator=Operator.equal, rhs=category)
        scoring.add_split_condition(condition=split_condition, split_point=category)

    return scoring.get_report()
