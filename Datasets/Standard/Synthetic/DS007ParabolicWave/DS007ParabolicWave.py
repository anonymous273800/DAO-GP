import numpy as np
from Plotter import Plotter


def DS007_ParabolicWave(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=5, weight_factor=0.5):
    """
    Generates a dataset where the first feature follows a parabolic sine wave,
    and additional features (if present) contribute noise.

    Parameters:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        noise (float): Noise level.
        lower_bound (float): Minimum value for X.
        upper_bound (float): Maximum value for X.
        stretch_factor (float): Controls frequency of the sine wave.
        weight_factor (float): Controls how much additional features influence y.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target values of shape (n_samples,).
    """
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Multi-feature dataset

    # Parabolic wave transformation on the first feature
    y = X[:, 0] ** 2 * np.sin(stretch_factor * X[:, 0])

    # Add contributions from additional features (if n_features > 1)
    if n_features > 1:
        y += weight_factor * X[:, 1:].sum(axis=1)  # Linear effect from other features

    # Add noise
    y += noise * np.random.randn(n_samples)

    return X, y


if __name__ == "__main__":
    X, y = DS007_ParabolicWave(200, 1, 0.2, -5, 5, stretch_factor=3, weight_factor=0.5)

    Plotter.plot_basic(X, y)
