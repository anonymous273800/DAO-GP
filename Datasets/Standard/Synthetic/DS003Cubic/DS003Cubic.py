import numpy as np
from sklearn.preprocessing import StandardScaler

from Plotter import Plotter

def DS003_Cubic(n_samples, n_features, noise, lower_bound, upper_bound):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = X[:, 0] ** 3 + noise * np.random.randn(n_samples)

    # # Standardize features and target.
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y

if __name__ == "__main__":
    X, y = DS003_Cubic(150, 1, .2, -5, 5)
    Plotter.plot_basic(X,y)
