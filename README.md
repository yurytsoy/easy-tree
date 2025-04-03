from easy_tree import DecisionTree

# eazy-tree

Easy-to-use implementation of decision tree algorithm for regression and classification. 
* Does not require data transform and can work natively with missing values.
* Supports lazy and eager data frames from [polars](https://pola.rs/).
* Prefers splits that lead to balanced tree.
* Single class for both classification and regression.
* Easy to extract and use rules corresponding to leaves.
* Save and load trained model.
* Niceties:
    * Supports training straight from the CSV-file. 
    * Simple library to build and apply logical expressions (in the `easy_tree/logic.py`).

The implementation does not provide many parameters on purpose. The main goal is to provide a robust
decision tree model without exhaustive tweaking of those parameters for (quite often) marginal gains.

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
tree.fit(data="tests/data/titanic_train.csv", y_true="Survived")  # will train a regression, as "Survived" is auto-detected as numerical column

tree = et.DecisionTree()
tree.fit(data="tests/data/titanic_train.csv", y_true="Embarked")  # will train a classification tree, as "Embarked" is auto-detected as String-valued column
```
