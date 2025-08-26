import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tree.base import DecisionTree  # Your custom implementation
from metrics import *

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, dtype="category")

# Set seed for reproducibility
np.random.seed(42)

# ---- Train and Plot Custom Decision Tree ----
print("\n--- Custom Decision Tree ---")
for criterion in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criterion)
    tree.fit(X, y)
    y_pred = tree.predict(X)
    tree.plot()

    print(f"Criterion: {criterion}")
    print("Accuracy:", accuracy(y_pred, y))
    for cls in y.unique():
        print(f"Class {cls} - Precision:", precision(y_pred, y, cls))
        print(f"Class {cls} - Recall:", recall(y_pred, y, cls))


# ---- Train and Plot scikit-learn Decision Tree ----
print("\n--- scikit-learn Decision Tree ---")
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X, y)

plt.figure(figsize=(15, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("scikit-learn Decision Tree (Gini)")
plt.show()