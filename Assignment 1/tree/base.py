"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNode:
    def __init__(self):
        # Splitting info
        self.attribute = None
        self.split_value = None
        self.is_real = None
        
        # Children
        self.left = None
        self.right = None
        
        # Leaf info
        self.is_leaf = False
        self.prediction = None

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        node = TreeNode()

        # Stopping conditions
        if depth == self.max_depth or y.nunique() == 1 or X.empty:
            node.is_leaf = True
            node.prediction = y.mean() if check_ifreal(y) else y.mode()[0]
            return node

        # Is our output real or discrete
        criterion = "mse" if check_ifreal(y) else ("entropy" if self.criterion == "information_gain" else "gini")

        # getting the best features and value to split upon 
        bestFeature, bstVal = opt_split_attribute(X, y, criterion)

        # Handle case where no valid split is found
        if bestFeature is None or bstVal is None:
            node.is_leaf = True
            node.prediction = y.mean() if check_ifreal(y) else y.mode()[0]
            return node
        
        node.attribute = bestFeature # Storing Node data
        node.split_value = bstVal
        node.is_real = check_ifreal(X[bestFeature])

        # Calling for the children
        (X_left, X_right), (y_left, y_right) = split_data(X, y, bestFeature, bstVal)

        # Handling if split is useless
        if len(y_left) == 0 or len(y_right) == 0:
            node.is_leaf = True
            node.prediction = y.mean() if check_ifreal(y) else y.mode()[0]
            return node

        node.left = self.build_tree(X_left, y_left, depth+1)
        node.right = self.build_tree(X_right, y_right, depth+1)

        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """ Function to train and construct the decision tree """
        self.root = self.build_tree(X, y, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """ Funtion to run the decision tree on test inputs """
        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        def traverse(node: TreeNode, data: pd.Series):
            while not node.is_leaf:
                val = data[node.attribute]
                if node.is_real:
                    if val <= node.split_value:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if val == node.split_value:
                        node = node.left
                    else:
                        node = node.right
            return node.prediction

        predictions = X.apply(lambda row : traverse(self.root, row), axis = 1)
        return predictions
        
    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def recurse(node: TreeNode, depth=0, prefix=""):
            indent = "    " * depth
            if node.is_leaf:
                print(f"{indent}{prefix}Predict: {node.prediction}")
                return
            
            condition = f"{node.attribute} <= {node.split_value}" if node.is_real else f"{node.attribute} == {node.split_value}"
            print(f"{indent}{prefix}If {condition}:")
            recurse(node.left, depth + 1, prefix="Y: ")
            recurse(node.right, depth + 1, prefix="N: ")

        if self.root is None:
            print("Tree is empty. Please call fit() first.")
        else:
            recurse(self.root)