import numpy as np
from sklearn.preprocessing import StandardScaler
from Plotter import Plotter


def DS002_Quadratic(n_samples, n_features, noise=0.1, lower_bound=-5, upper_bound=5):
    """Generate a quadratic dataset: y = x^2 + noise."""
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = X[:, 0] ** 2 + noise * np.random.randn(n_samples)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    return X, y

# def DS002_Quadratic(n_samples, n_features, noise=0.1, lower_bound=-5, upper_bound=5):
#     """Generate a quadratic dataset: y = sum of x_i^2 + noise."""
#     X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
#     y = np.sum(X ** 2, axis=1) + noise * np.random.randn(n_samples)
#     X = StandardScaler().fit_transform(X)
#     y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
#     return X, y


if __name__ == "__main__":
    n_samples = 300
    n_features = 1
    noise = .1
    lower_bound = -2
    upper_bound = 2
    X, y = DS002_Quadratic(n_samples, n_features, noise, lower_bound, upper_bound)
    Plotter.plot_basic(X,y)