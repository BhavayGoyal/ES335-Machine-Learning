import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values

# Their are a few values with '?' as horsepower, so set them to .nan and remove those rows
data.replace({'horsepower': '?'}, np.nan, inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'])
data.dropna(inplace=True)
# 305 of 398 rows have different car names, so the columns seems to be redundant
data.drop(columns=['car name'], inplace=True)

# Compare the performance of your model with the decision tree module from scikit learn
X = data.drop(columns=['mpg'])
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our decision Tree
tree = DecisionTree("information_gain", max_depth=5)
tree.fit(X_train, y_train)
y_pred_custom = tree.predict(X_test)
print("Our Custom Decision Tree:")
print("MAE:", mae(y_test, y_pred_custom))        
print("MSE:", rmse(y_test, y_pred_custom))

# Evaluating on sklearn decision tree
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=42, max_depth=5)
reg.fit(X_train, y_train)
y_pred_sklearn = reg.predict(X_test)
print("Scikit-learn Decision Tree:")
print("MAE:", mae(y_test, y_pred_sklearn))
print("MSE:", rmse(y_test, y_pred_sklearn))

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='True', marker='o')
plt.plot(y_pred_custom.values, label='Custom Tree', marker='x')
plt.plot(y_pred_sklearn, label='Sklearn Tree', marker='s')
plt.title('Comparison of Predictions')
plt.xlabel('Sample Index')
plt.ylabel('MPG')
plt.legend()
plt.show()