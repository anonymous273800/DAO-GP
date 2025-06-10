import numpy as np
from Plotter import Plotter

def DS03_Exponential(n_samples, n_features, noise, lower_bound, upper_bound):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.exp(X[:, 0]) + noise * np.random.randn(n_samples)

    return X, y

if __name__ == "__main__":
    X, y = DS03_Exponential(150, 1, .2, -2,2)
    Plotter.plot_basic(X,y)