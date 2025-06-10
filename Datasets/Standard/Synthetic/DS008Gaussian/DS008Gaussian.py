import numpy as np
from Plotter import Plotter

def DS008_Gaussian(n_samples, n_features, noise, lower_bound, upper_bound, sigma=1.0, weight_factor=0.5):
    """
    Generates a Gaussian-shaped dataset with optional multi-feature influence.

    Parameters:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        noise (float): Noise level.
        lower_bound (float): Minimum value for X.
        upper_bound (float): Maximum value for X.
        sigma (float): Controls the spread of the Gaussian function.
        weight_factor (float): Controls how much additional features influence y.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target values of shape (n_samples,).
    """
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Multi-feature dataset

    # Multi-dimensional Gaussian function (sum of squared features)
    y = np.exp(-np.sum(X ** 2, axis=1) / (2 * sigma ** 2))

    # Add contributions from additional features (if n_features > 1)
    if n_features > 1:
        y += weight_factor * X[:, 1:].sum(axis=1)

    # Add noise
    y += noise * np.random.randn(n_samples)

    return X, y


if __name__ == "__main__":
    X, y = DS008_Gaussian(150, 1, 0.1, -2, 2, sigma=1, weight_factor=1.3)

    Plotter.plot_basic(X, y)
