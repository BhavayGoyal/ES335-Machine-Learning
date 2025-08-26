import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 1  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
def get_data(N, P, type, disFeatureCount=5):
    X, y = None, None # real real, real dis, dis dis, dis real
    if type == 0:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
    elif type == 1:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(disFeatureCount, size=N), dtype="category")
    elif type == 2:
        X = pd.DataFrame({i: pd.Series(np.random.randint(disFeatureCount, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(disFeatureCount, size=N), dtype="category")
    else:
        X = pd.DataFrame({i: pd.Series(np.random.randint(disFeatureCount, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))
    
    return X, y

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def run_experiment(N, P, type):
    fit_times = []
    pred_times = []
    for iter in range(num_average_time):
        X, y = get_data(N, P, type)
        tree = DecisionTree("gini_index", max_depth=5)
        X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
        y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
        
        start_time = time.time()
        tree.fit(X_train, y_train)
        end_time = time.time()
        fit_times.append(end_time-start_time)

        start_time = time.time()
        tree.predict(X_test)
        end_time = time.time()
        pred_times.append(end_time-start_time)
    return np.mean(fit_times), np.std(fit_times), np.mean(pred_times), np.std(pred_times)

# Plot results function
def plot_results(results, xlabel, ylabel, title, legend_labels, x_values):
    plt.figure(figsize=(12, 8))
    for i, (means, stds) in enumerate(results):
        means = np.array(means)
        stds = np.array(stds)
        x = np.arange(len(means))
        plt.errorbar(x, means, yerr=stds, label=legend_labels[i], capsize=3, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(x_values)), x_values)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    types = [0, 1, 2, 3]
    type_names = ["real-real", "real-discrete", "discrete-discrete", "discrete-real"]

    # Vary N (number of samples) keeping P fixed
    P_fixed = 10
    N_values = [10, 50, 100, 200, 500, 1000, 10000]
    fit_results_N = []
    pred_results_N = []

    for t in types:
        fit_means, fit_stds, pred_means, pred_stds = [], [], [], []
        print(f"Running experiments for type={t} (varying N)...")
        for N in N_values:
            mean_fit, std_fit, mean_pred, std_pred = run_experiment(N, P_fixed, t)
            fit_means.append(mean_fit)
            fit_stds.append(std_fit)
            pred_means.append(mean_pred)
            pred_stds.append(std_pred)
        fit_results_N.append((fit_means, fit_stds))
        pred_results_N.append((pred_means, pred_stds))

    plot_results(fit_results_N, xlabel="Number of Samples (N)", ylabel="Fit Time (seconds)",
                 title=f"Decision Tree Fit Time vs N (P={P_fixed}) for different data types", legend_labels=type_names, x_values=N_values)

    plot_results(pred_results_N, xlabel="Number of Samples (N)", ylabel="Predict Time (seconds)",
                 title=f"Decision Tree Predict Time vs N (P={P_fixed}) for different data types", legend_labels=type_names, x_values=N_values)

    # Vary P (number of features) keeping N fixed
    N_fixed = 1000
    P_values = [5, 10, 20, 30, 40, 50, 100]
    fit_results_P = []
    pred_results_P = []

    for t in types:
        fit_means, fit_stds, pred_means, pred_stds = [], [], [], []
        print(f"Running experiments for type={t} (varying P)...")
        for P in P_values:
            mean_fit, std_fit, mean_pred, std_pred = run_experiment(N_fixed, P, t)
            fit_means.append(mean_fit)
            fit_stds.append(std_fit)
            pred_means.append(mean_pred)
            pred_stds.append(std_pred)
        fit_results_P.append((fit_means, fit_stds))
        pred_results_P.append((pred_means, pred_stds))

    plot_results(fit_results_P, xlabel="Number of Features (P)", ylabel="Fit Time (seconds)",
                 title=f"Decision Tree Fit Time vs P (N={N_fixed}) for different data types", legend_labels=type_names, x_values=P_values)

    plot_results(pred_results_P, xlabel="Number of Features (P)", ylabel="Predict Time (seconds)",
                 title=f"Decision Tree Predict Time vs P (N={N_fixed}) for different data types", legend_labels=type_names, x_values=P_values)

if __name__ == "__main__":
    main()