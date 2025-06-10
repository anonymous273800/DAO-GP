import numpy as np
from sklearn.preprocessing import StandardScaler

from Datasets.Standard.Synthetic.DS002Quadratic.DS002Quadratic import DS002_Quadratic
from Plotter import Plotter


def DS002_Quadratic_Normalized(n_samples, n_features, noise=0.1, lower_bound=-5, upper_bound=5, y_shift=0):
    """Generate a quadratic dataset: y = x^2 + noise."""
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = X[:, 0] ** 2 + noise * np.random.randn(n_samples)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel() + y_shift
    return X, y


def DS002_Quadratic_Not_Normalized(n_samples, n_features, noise=0.1, lower_bound=-5, upper_bound=5, y_shift=0):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = X[:, 0] ** 2 + noise * np.random.randn(n_samples) + y_shift
    return X, y


if __name__ == "__main__":
    n_samples = 300
    n_features = 1
    noise = .1
    lower_bound = -2
    upper_bound = 2
    y_shift = 0
    X, y = DS002_Quadratic(n_samples, n_features, noise, lower_bound, upper_bound, y_shift)
    Plotter.plot_basic(X,y)