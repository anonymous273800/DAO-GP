import numpy as np
from sklearn.preprocessing import StandardScaler
from Plotter import Plotter


def get_sin_dataset(n_samples):
    """Generate a noisy sine wave dataset."""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.2 * np.random.randn(n_samples)
    return X, y

def DS001_Sinusoidal(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + noise * np.random.randn(n_samples)

    # Standardize features and target.
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y

def DS001_Sinusoidal_Not_Normalized(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + noise * np.random.randn(n_samples)

    return X, y

# def DS001_Sinusoidal(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1):
#     X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
#
#     # Each dimension contributes a sinusoidal signal
#     y = np.sum(np.sin((2 * np.pi / stretch_factor) * X), axis=1)
#
#     # Add Gaussian noise
#     y += noise * np.random.randn(n_samples)
#
#     # Standardize
#     X = StandardScaler().fit_transform(X)
#     y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
#
#     return X, y


if __name__ == "__main__":
    n_samples = 300
    n_features = 1
    noise = .1
    lower_bound = -2
    upper_bound = 2
    stretch_factor=1
    X, y = DS001_Sinusoidal(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=stretch_factor)  # Stretching the sine wave
    Plotter.plot_basic(X,y)