import numpy as np
from Plotter import Plotter


def DS06_Piecewise(n_samples, n_features, noise, lower_bound, upper_bound):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.piecewise(X[:, 0], [X[:, 0] < 0, X[:, 0] >= 0], [lambda x: x ** 2, lambda x: np.sin(5 * x)]) + noise * np.random.randn(n_samples)

    return X, y

if __name__ == "__main__":
    X, y = DS06_Piecewise(150, 1, 0.2, -2, 2)

    Plotter.plot_basic(X,y)

