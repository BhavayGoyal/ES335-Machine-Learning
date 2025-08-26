import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tree.utils import kfold_split_indices

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
X = pd.DataFrame(X); y = pd.Series(y);

# Question 2 a)
# Split the dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
tree.plot()
print("Criteria :", "information_gain")
print("Accuracy: ", accuracy(y_hat, y_test))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y_test, cls))
    print("Recall: ", recall(y_hat, y_test, cls))

# Question 2 b)

outer_scores = []
for outer_train_indices, outer_val_indices in kfold_split_indices(len(X), k=5):
    outer_train_X, outer_train_y = X.iloc[outer_train_indices], y.iloc[outer_train_indices]
    outer_val_X, outer_val_y = X.iloc[outer_val_indices], y.iloc[outer_val_indices]
    
    bestDepth = None; bstScore = -np.inf
    for depth in [1, 2, 3, 4, 5, 6, 7, 8]:
        scores = []

        for inner_train_indices, inner_val_indices in kfold_split_indices(len(outer_train_X), k=5):
            inner_train_X, inner_train_y = outer_train_X.iloc[inner_train_indices], outer_train_y.iloc[inner_train_indices]
            inner_val_X, inner_val_y = outer_train_X.iloc[inner_val_indices], outer_train_y.iloc[inner_val_indices]

            tree = DecisionTree(criterion="information_gain", max_depth=depth)
            tree.fit(inner_train_X, inner_train_y)
            predictions = tree.predict(inner_val_X)
            acc = (predictions == inner_val_y).mean()
            scores.append(acc)

        avg_score = np.mean(scores)
        if avg_score > bstScore:
            bstScore = avg_score; bestDepth = depth
    
    final_tree = DecisionTree(criterion="information_gain", max_depth=bestDepth)
    final_tree.fit(outer_train_X, outer_train_y)
    outer_preds = final_tree.predict(outer_val_X)
    outer_acc = (outer_preds == outer_val_y).mean()

    print(f"Best depth, current accuracy for current fold = {bestDepth}, {outer_acc}")
    outer_scores.append(outer_acc)

print("Precision", np.mean(outer_scores))