import numpy as np
import matplotlib.pyplot as plt
from Plotter import Plotter
from sklearn.preprocessing import StandardScaler
def DS10_DoubleGaussian(n_samples, n_features, noise, lower_bound, upper_bound,
                        sigma1=1.0, sigma2=1.0, peak_offset=2.0, mix_ratio=0.5, weight_factor=0.5):
    """
    Generates a dataset with two distinct Gaussian peaks.

    Parameters:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        noise (float): Noise level.
        lower_bound (float): Minimum value for X.
        upper_bound (float): Maximum value for X.
        sigma1 (float): Spread of the first Gaussian.
        sigma2 (float): Spread of the second Gaussian.
        peak_offset (float): Offset of the second Gaussian.
        mix_ratio (float): Mixing ratio between the two Gaussians.
        weight_factor (float): Contribution of additional features.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target values of shape (n_samples,).
    """
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Multi-feature dataset

    # Use only X[:, 0] for Gaussian shape
    gaussian1 = np.exp(-((X[:, 0]) ** 2) / (2 * sigma1 ** 2))  # Peak at 0
    gaussian2 = np.exp(-((X[:, 0] - peak_offset) ** 2) / (2 * sigma2 ** 2))  # Peak at peak_offset

    # Weighted sum of both Gaussians
    y = mix_ratio * gaussian1 + (1 - mix_ratio) * gaussian2

    # Add contributions from additional features (if n_features > 1)
    if n_features > 1:
        y += weight_factor * X[:, 1:].sum(axis=1)

    # Add noise
    y += noise * np.random.randn(n_samples)

    # Standardize features and target.
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y


if __name__ == "__main__":
    X, y = DS10_DoubleGaussian(150, 1, 0.1, -5, 5, sigma1=0.5, sigma2=0.5, peak_offset=3.0, mix_ratio=0.6)

    Plotter.plot_basic(X,y)
