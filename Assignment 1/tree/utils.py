import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """Return True if series is real-valued (float or high-cardinality int), False for discrete/categorical."""
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y) and not pd.api.types.is_categorical_dtype(y):
        # Allow high-cardinality int columns to be treated as real
        return y.nunique() > 15 and (y.nunique()/len(y)) > 0.15
    return False


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """ Perform one-hot encoding on all categorical (discrete) columns of DataFrame. """
    X_new = X.copy()
    
    for col in X.columns:
        if not check_ifreal(X[col]):  # discrete column
            dummies = pd.get_dummies(X_new[col], prefix=col, drop_first=False)
            X_new = pd.concat([X_new.drop(columns=[col]), dummies], axis=1)
    
    return X_new

def mse(Y: pd.Series) -> float:
    """ Function to calculate the mean squared error """
    mean = Y.mean()
    return np.mean((Y-mean)**2)

def entropy(Y: pd.Series) -> float:
    """ Function to calculate the entropy """
    counts = Y.value_counts(normalize=True) # Count is nothing but the probabilities too 
    counts = counts[counts > 0]
    return -np.sum(counts*np.log2(counts))


def gini_index(Y: pd.Series) -> float:
    """ Function to calculate the gini index """
    counts = Y.value_counts(normalize=True)
    return 1 - np.sum(counts*counts)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str, split_point) -> float:
    """ Function to calculate the information gain using criterion (entropy, gini index or MSE) """
    impurity_func = None
    if criterion == "entropy":
        impurity_func = entropy
    elif criterion == "gini":
        impurity_func = gini_index
    elif criterion == "mse":
        impurity_func = mse
    else:
        raise ValueError("Invalid criterion. Choose from 'entropy', 'gini', or 'mse'.")
    
    mask = None
    if check_ifreal(attr):
        mask = attr <= split_point
    else:
        mask = attr == split_point
    
    left = Y[mask]
    right = Y[~mask]

    if len(left) == 0 or len(right) == 0:
        return -np.inf

    total = impurity_func(Y)
    weighted = len(left)/len(Y)*impurity_func(left) + len(right)/len(Y)*impurity_func(right)

    return total - weighted

def opt_split_real_attribute(X_col: pd.Series, y: pd.Series, criterion: str):
    sorted_idx = X_col.argsort()
    X_sorted = X_col.iloc[sorted_idx].reset_index(drop=True)
    y_sorted = y.iloc[sorted_idx].reset_index(drop=True)

    unique_vals = X_sorted.unique()
    if len(unique_vals) == 1:
        return None, -np.inf  # No split possible

    n = len(y_sorted)
    best_split = None
    max_gain = -np.inf

    if criterion == "mse":
        # Regression case: real-valued y
        prefix_sum = np.cumsum(y_sorted)
        prefix_sq_sum = np.cumsum(y_sorted ** 2)

        total_sum = prefix_sum.iloc[-1]
        total_sq_sum = prefix_sq_sum.iloc[-1]
        for i in range(1, n):
            if X_sorted[i] == X_sorted[i - 1]:
                continue  # Can't split on identical feature values

            left_n = i
            right_n = n - i

            left_sum = prefix_sum[i - 1]
            right_sum = total_sum - left_sum

            left_sq_sum = prefix_sq_sum[i - 1]
            right_sq_sum = total_sq_sum - left_sq_sum

            left_var = (left_sq_sum / left_n) - (left_sum / left_n) ** 2
            right_var = (right_sq_sum / right_n) - (right_sum / right_n) ** 2

            weighted_var = (left_n / n) * left_var + (right_n / n) * right_var
            info_gain = ((total_sq_sum / n) - (total_sum / n) ** 2) - weighted_var

            split_val = (X_sorted[i] + X_sorted[i - 1]) / 2
            if info_gain > max_gain:
                max_gain = info_gain
                best_split = split_val

    else:
        # Encode classes as integers
        classes = list(dict.fromkeys(y_sorted))   # preserves order
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_enc = np.array([class_to_idx[v] for v in y_sorted])

        k, n = len(classes), len(y_enc)
        left, right = np.zeros(k, int), np.array([np.sum(y_enc==i) for i in range(k)])

        def impurity(c):
            total = c.sum()
            if total == 0: return 0.0
            p = c / total
            return -np.sum(p*np.log2(p+1e-12)) if criterion=="entropy" else 1 - np.sum(p**2)

        total_imp = impurity(right)

        for i in range(1, n):
            if y_enc[i] == y_enc[i-1]:
                continue
            c = y_enc[i-1]
            left[c] += 1; right[c] -= 1
            gain = total_imp - (i/n)*impurity(left) - ((n-i)/n)*impurity(right)
            if gain > max_gain:
                max_gain, best_split = gain, (X_sorted[i-1]+X_sorted[i])/2

    return best_split, max_gain

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str):
    """
    Find the optimal attribute and split value among all features.
    Uses prefix optimization for real-valued features.
    """
    max_gain = -np.inf
    bestFeature = None
    bestVal = None

    for feature in X.columns:
        attr = X[feature]

        if check_ifreal(attr):
            split_val, info_gain = opt_split_real_attribute(attr, y, criterion)
            if info_gain > max_gain:
                max_gain = info_gain
                bestFeature = feature
                bestVal = split_val
        else:
            # For discrete features, test all unique values directly (no optimization needed)
            for val in attr.unique():
                info_gain = information_gain(y, attr, criterion, val)
                if info_gain > max_gain:
                    max_gain = info_gain
                    bestFeature = feature
                    bestVal = val

    return bestFeature, bestVal

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """ Funtion to split the data according to an attribute. return: splitted data(Input and output) """
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        mask = X[attribute] <= value
    else:
        mask = X[attribute] == value
    return (X[mask], X[~mask]), (y[mask], y[~mask])

def kfold_split_indices(n_samples:int, k=5, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    fold_sizes = (n_samples // k) * np.ones(k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    current = 0
    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        yield train_indices, val_indices
        current += fold_size