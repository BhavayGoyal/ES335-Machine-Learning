## Runtime Complexity Analysis from `experiments.py`

For Fitting, the time complexity for fitting a dataset into a Decision Tree is **O(n * m * log(n))** where **n** is the number of rows in the dataset and **m** is the number of features used for fitting.

For predicting, the time complexity is of the order **O(log(n))** or **O(h)**, where **h** is the maximum depth of the decision tree, whichever is lower.

Here is a plot from our analysis:

![Decision Tree Fit Time vs N (P=10)](Images/Decision_Tree_Fit_Time_vs_N_(P=10)_for_different_data_types.png)
![Decision Tree Predict Time vs N (P=10)](Images/Decision_Tree_Predict_Time_vs_N_(P=10)_for_different_data_types.png)
![Decision Tree Fit Time vs P (N=1000)](Images/Decision_Tree_Fit_Time_vs_P_(N=1000)_for_different_data_types.png)
![Decision Tree Predict Time vs P (N=1000)](Images/Decision_Tree_Predict_Time_vs_P_(N=1000)_for_different_data_types.png)


The plot for both fitting and predicting is matching the shape of theoretical graphs in essence.

