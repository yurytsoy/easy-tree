from __future__ import annotations

from dataclasses import dataclass
import polars as pl


class Operator:
    less = "<"
    less_or_equal = "<="
    greater = ">"
    greater_or_equal = ">="
    equal = "=="
    not_equal = "!="


class BaseExpression:
    """
    Base class for all logical expressions
    """

    @property
    def column(self) -> str:
        return self.to_polars().meta.output_name()

    def to_polars(self) -> pl.Expr:
        pass

    def apply(self, data: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        tmp_name = "__expr__"
        res = data.with_columns(self.to_polars().alias(tmp_name)).select(tmp_name)
        if isinstance(res, pl.LazyFrame):
            res = res.collect()
        return res.to_series()

    def _serialize(self) -> dict:
        pass

    @staticmethod
    def _deserialize(data: dict) -> BaseExpression:
        pass

    def serialize(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "data": self._serialize()
        }

    @staticmethod
    def deserialize(data: dict | None) -> BaseExpression | None:
        """
        Deserialize BaseExpression instance, which was serialized by `BaseExpression.serialize`.
        """

        if data is None:
            return None

        functions = {  # deserialization functions depending on the object type. See also `BaseExpression.serialize`.
            AtomicExpression.__name__: AtomicExpression._deserialize,
            AndExpression.__name__: AndExpression._deserialize,
            OrExpression.__name__: OrExpression._deserialize,
            NotExpression.__name__: NotExpression._deserialize,
        }

        if data["type"] in functions:
            return functions[data["type"]](data["data"])

        raise NotImplementedError(data["type"])


@dataclass
class AtomicExpression(BaseExpression):
    """
    Leaf class: Atomic Boolean expression (e.g., A > 5)
    """

    colname: str
    operator: Operator
    rhs: int | float | str | None

    def to_polars(self) -> pl.Expr:
        """
        Returns Polars expression for condition. The expression result is Boolean.

        Args
        ----
        propagate_null : bool, default=True
            Defines how empty values are handled. Polars propagates null and NaNs during comparison.
            This can result is having undefined outcomes when combining conditions.
            In order to avoid nulls in the comparison results we can force boolean outcome when data contains
            missing values.
        """

        if self.operator == Operator.less:
            return pl.col(self.colname).lt(self.rhs)

        if self.operator == Operator.less_or_equal:
            return pl.col(self.colname).le(self.rhs)

        if self.operator == Operator.greater:
            return pl.col(self.colname).gt(self.rhs)

        if self.operator == Operator.greater_or_equal:
            return pl.col(self.colname).ge(self.rhs)

        if self.operator == Operator.equal:
            if self.rhs is not None:
                return pl.col(self.colname).eq(self.rhs)
            else:
                return pl.col(self.colname).is_null()

        if self.operator == Operator.not_equal:
            if self.rhs is not None:
                return pl.col(self.colname).ne(self.rhs)
            else:
                return pl.col(self.colname).is_not_null()

    def not_(self) -> AtomicExpression:
        not_operator = {
            Operator.less: Operator.greater_or_equal,
            Operator.less_or_equal: Operator.greater,
            Operator.greater: Operator.less_or_equal,
            Operator.greater_or_equal: Operator.less,
            Operator.equal: Operator.not_equal,
            Operator.not_equal: Operator.equal,
        }
        return AtomicExpression(colname=self.colname, operator=not_operator[self.operator], rhs=self.rhs)

    def __repr__(self) -> str:
        return f"{self.colname} {self.operator} {self.rhs}"

    def _serialize(self) -> dict:
        return {
            "colname": self.colname,
            "operator": self.operator,
            "rhs": self.rhs,
        }

    @staticmethod
    def _deserialize(data: dict) -> AtomicExpression:
        return AtomicExpression(**data)


# Composite classes for logical connectives
class AndExpression(BaseExpression):
    def __init__(
        self,
        left: BaseExpression,
        right: BaseExpression,
    ):
        self.left = left
        self.right = right

    def to_polars(self) -> pl.Expr:
        return self.left.to_polars() & self.right.to_polars()

    def __repr__(self) -> str:
        return f"({self.left}) AND ({self.right})"

    def _serialize(self) -> dict:
        return {
            "left": self.left.serialize() if self.left is not None else None,
            "right": self.right.serialize() if self.right is not None else None,
        }

    @staticmethod
    def _deserialize(data: dict) -> AndExpression:
        return AndExpression(
            left=BaseExpression.deserialize(data["left"]),
            right=BaseExpression.deserialize(data["right"]),
        )


class OrExpression(BaseExpression):
    def __init__(
        self,
        left: BaseExpression,
        right: BaseExpression,
    ):
        self.left = left
        self.right = right

    def to_polars(self) -> pl.Expr:
        return self.left.to_polars() | self.right.to_polars()

    def __repr__(self) -> str:
        return f"({self.left}) OR ({self.right})"

    def _serialize(self) -> dict:
        return {
            "left": self.left.serialize() if self.left is not None else None,
            "right": self.right.serialize() if self.right is not None else None,
        }

    @staticmethod
    def _deserialize(data: dict) -> OrExpression:
        return OrExpression(
            left=BaseExpression.deserialize(data["left"]),
            right=BaseExpression.deserialize(data["right"]),
        )


class NotExpression(BaseExpression):
    def __init__(
        self,
        expr: BaseExpression,
    ):
        self.expr = expr

    def not_(self) -> BaseExpression:
        return self.expr

    def to_polars(self) -> pl.Expr:
        return ~self.expr.to_polars()

    def __repr__(self) -> str:
        return f"NOT ({self.expr})"

    def _serialize(self) -> dict:
        return {
            "expr": self.expr.serialize() if self.expr is not None else None
        }

    @staticmethod
    def _deserialize(data: dict) -> NotExpression:
        return NotExpression(
            expr=BaseExpression.deserialize(data["expr"])
        )


class ExpressionBuilder(BaseExpression):
    """
    Builder pattern for ... building logical expressions and conditions.
    """

    def __init__(self, start: BaseExpression | None = None):
        self.current = start

    def and_(self, other: BaseExpression) -> ExpressionBuilder:
        if self.current is None:
            raise RuntimeError()

        self.current = AndExpression(self.current, other)
        return self

    def or_(self, other: BaseExpression) -> ExpressionBuilder:
        if self.current is None:
            raise RuntimeError()

        self.current = OrExpression(self.current, other)
        return self

    def not_(self) -> ExpressionBuilder:
        if self.current is None:
            raise RuntimeError()

        if hasattr(self.current, "not_"):
            self.current = self.current.not_()
        else:
            # Proper negation of AND and OR is not implemented yet.
            #   Also, it can make resulting leaf conditions look more confusing
            self.current = NotExpression(self.current)
        return self

    def to_polars(self) -> pl.Expr:
        if self.current is None:
            raise RuntimeError()

        return self.current.to_polars()

    def apply(self, data: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        if self.current is None:
            raise RuntimeError()

        return self.current.apply(data)

    def __repr__(self) -> str:
        if self.current is None:
            return ""

        return str(self.current)
