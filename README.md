# easy-tree

Decision tree algorithm implementation for regression and classification with minimal boilerplate code. 
* Does not require data transform and can work natively with missing values.
* Supports lazy and eager data frames from [polars](https://pola.rs/).
* Prefers splits that lead to balanced tree.
* Single class for both classification and regression.
* Easy to extract and use rules corresponding to leaves.
* Save and load trained model.
* Little niceties:
    * Supports training straight from the CSV-file. One-line training is also possible.
    * Simple library to build and apply logical expressions (in the `easy_tree/logic.py`).

The implementation does not provide many parameters on purpose. The main goal is to provide a robust
decision tree model without exhaustive tweaking of those parameters for quite often marginal gains
and questionable stability.

The API resembles that of [scikit-learn](https://scikit-learn.org/stable/) and should be familiar for most people.

# Quick start

## Training a regression tree model

```python
import easy_tree as et
import polars as pl


# load lazy data frame and extract target as polars series.
df = pl.scan_csv("tests/data/titanic_train.csv")
y_true = df.select("Fare").collect().to_series()

# train a tree, no data preprocessing is required!
tree = et.DecisionTree()
tree.fit(data=df, y_true=y_true)

# compute predictions and MSE.
pred = tree.predict(df)
mse = (pred - y_true).pow(2).mean()
print(f"MSE = {mse}")
```

Currently, the following parameters are exposed for tuning:
* `max_depth` -- maximal tree depth. Default value is 4.
* `min_leaf_size` -- minimal number of samples in the leaf. Default is 10.

## Training a classification tree

If target series has dtype `String` then classification model is trained.

```python
import easy_tree as et
import polars as pl


# load data
df = pl.scan_csv("tests/data/titanic_train.csv")
y_true = df.select("Survived").collect().to_series().cast(str)  # target has dtype `String`

# train a tree using the same class
tree = et.DecisionTree()
tree.fit(data=df, y_true=y_true)

# compute prediction and accuracy
pred = tree.predict(df)  # prediction has dtype `String`
accuracy = (pred == y_true).mean()
print(f"Accuracy = {accuracy}")
```

## Feature importances

Feature importances are available via the `feature_importances_` property.
```python
# feature importances
print({name: f"{importance:.3f}" for name, importance in tree.feature_importances_.items()})
# {'Sex': '0.631', 'Pclass': '0.208', 'Cabin': '0.093', 'Embarked': '0.034', 'Parch': '0.015', 'Age': '0.012', 'Fare': '0.007'}
```

## Working with tree leaves

The trained `DecisionTree` object has a property `leaves` that contains list of, well, tree leaves.
All tree nodes can be accessed via the method `get_nodes`.

Each leaf has a property `full_condition` that represent "path" from the root of the tree to that leaf.

An example below prints out conditions for the classification tree trained on the Titanic data from the example above.

```python
print("\n".join(str(leaf.full_condition) for leaf in tree.leaves))
# ((Sex != male) AND (NOT ((Pclass > 2.0) AND (Pclass != None)))) AND (NOT ((Fare > 49.5042) AND (Fare != None)))
# ((Sex != male) AND (NOT ((Pclass > 2.0) AND (Pclass != None)))) AND ((Fare > 49.5042) AND (Fare != None))
# ((Sex != male) AND ((Pclass > 2.0) AND (Pclass != None))) AND (Embarked != S)
# ((Sex != male) AND ((Pclass > 2.0) AND (Pclass != None))) AND (Embarked == S)
# ((Sex == male) AND (Cabin != None)) AND (NOT ((Age > 37.0) AND (Age != None)))
# ((Sex == male) AND (Cabin != None)) AND ((Age > 37.0) AND (Age != None))
# ((Sex == male) AND (Cabin == None)) AND (NOT ((Parch > 0.0) AND (Parch != None)))
# ((Sex == male) AND (Cabin == None)) AND ((Parch > 0.0) AND (Parch != None))
```

## Checking target statistics for leafs

Each node has attribute `target_stats` which is computed during training from training data and has the following fields:
* `score_reduction` -- how much the node's split contributed to the score reduction
* `mean` -- [regression] mean value of the target in the node
* `var` -- [regression] target variance in the node
* `distr` -- [classification] dictionary in the format `{"class_name": count}` for target classes

Example for regression (target: "Fare"):
```python
print(tree.leaves[0].target_stats)
# TargetStats(score_reduction=307.79988432965, mean=75.86526103896104, var=4915.438094119127, distr=None)
```

Example for binary classification (target: "Survived", cast to `str`):
```python
print(tree.leaves[0].target_stats)
# TargetStats(score_reduction=0.029410696893090726, mean=None, var=None, distr={'0': 7, '1': 79})
```

Example for non-binary classification (target: "Embarked"):
```python
print(tree.leaves[0].target_stats)
# TargetStats(score_reduction=0.11205789293238799, mean=None, var=None, distr={'Q': 24, 'C': 7, 'S': 6})
```

## Applying leaf conditions

Each leaf can be applied to the data in order to compute a mask corresponding to that leaf.

```python
df = pl.scan_csv("tests/data/titanic_train.csv")
mask = tree.leaves[0].full_condition.apply(df)  # boolean mask corresponding to the leaf samples.
masked_df = df.filter(mask)  # subset of the data samples, that "belong" to the leaf.
```

## Save and load model as json

Use methods `save` and `load` in order to save and load the decision tree

```python
tree.save("tree.json")
tree2 = et.DecisionTree.load("tree.json")
assert all(tree.predict(df) == tree2.predict(df))  # `tree2` produces the same predictions.
```

## Train straight from CSV-file

Decision tree can be trained by providing filename and name of the target column as `data` and `y_true` respectively.

In that case the type of the model (regression or classification) is defined based on the target-column type, detected by polars. 

```python
tree = et.DecisionTree()
tree.fit(data="tests/data/titanic_train.csv", y_true="Fare")  # will train a regression, as "Fare" is auto-detected as numerical column.

# -----------------

tree = et.DecisionTree()
tree.fit(data="tests/data/titanic_train.csv", y_true="Survived")  # will train a regression, as "Survived" is auto-detected as numerical column

# -----------------

tree = et.DecisionTree()
tree.fit(data="tests/data/titanic_train.csv", y_true="Embarked")  # will train a classification tree, as "Embarked" is auto-detected as String-valued column
```

One-liner is also possible:
```python
tree = et.DecisionTree().fit(data="tests/data/titanic_train.csv", y_true="Embarked")
```

## Small tool to handle logical expressions

Defined in the `easy_tree/logic.py` and supports:
* Atomic expression, such as `A <= 5` and `C != "foo"`.
* Connectives `AND`, `OR`, and `NOT`.
* Comparison with `None`, so that `bar != None` and `bar == None` are possible.
* `Serialization` and `deserialization` of expressions. 
* Making compound expressions using class `ExpressionBuilder`.
* Export to `polars.Expr` class which can be applied to the polars data frame.

Some examples (more can be found in the `tests/test_logic.py`):
```python
from easy_tree.logic import AtomicExpression, AndExpression, OrExpression, NotExpression, ExpressionBuilder, Operator

atomic = AtomicExpression(colname="Age", operator=Operator.greater, rhs=30)  # Age > 30
atomic_2 = AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="C")  # AtomicExpression(colname="Embarked", operator=Operator.equal, rhs="C") 
and_expr = AndExpression(left=atomic, right=atomic_2)  # (Age > 30) AND (Embarked == C)
or_expr = OrExpression(left=and_expr, right=AtomicExpression(colname="Age", operator=Operator.not_equal, rhs=None))  # ((Age > 30) AND (Embarked == C)) OR (Age != None)
expr_builder = (ExpressionBuilder(atomic)
                .and_(atomic_2)
                .or_(AtomicExpression(colname="Age", operator=Operator.not_equal, rhs=None))
                .not_())
# use property `current` to access the resulting expression
expr_builder.current  # NOT (((Age > 30) AND (Embarked == C)) OR (Age != None))
```

NOT is a bit special, because it can affect the comparison operator. `AtomicExpression` and `NotExpression` have method `not_()`
that handle the `negation` and can avoid double `negation`.
```python
not_expr_1 = NotExpression(atomic_2)  # NOT (Embarked == C)
not_expr_1_1 = atomic_2.not_()  # Embarked != C
not_expr_1_2 = ExpressionBuilder(atomic_2).not_().current  # Embarked != C
not_expr_1_3 = NotExpression(not_expr_1)  # NOT (NOT (Embarked == C))
not_expr_1_4 = not_expr_1.not_()  # Embarked == C
not_expr_1_5 = ExpressionBuilder(atomic_2).not_().not_().current  # Embarked == C
not_expr_2 = NotExpression(atomic)  # NOT (Age > 30)
not_expr_2_1 = atomic.not_()  # Age <= 30
not_expr_2_2 = ExpressionBuilder(atomic).not_().current  # Age <= 30
```

Currently NOT does not expand for AND and OR connectives (De Morgan's laws are not implemented).
The following expressions are equivalent:  
```python
not_expr_3 = NotExpression(and_expr)  # NOT ((Age > 30) AND (Embarked == C)), 
not_expr_3_1 = OrExpression(left=and_expr.left.not_(), right=and_expr.right.not_())  # (Age <= 30) OR (Embarked != C)
```
