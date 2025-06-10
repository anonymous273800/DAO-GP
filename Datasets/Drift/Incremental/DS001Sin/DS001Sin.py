import numpy as np
import matplotlib.pyplot as plt
from Plotter import Plotter

def DS001_Sinusoidal(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1, y_shift=0):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + noise * np.random.randn(n_samples) + y_shift
    y = y.ravel()
    return X, y


if __name__ == "__main__":

    # Define parameters for the first dataset (Before Drift)
    n_samples_1 = 300
    n_features = 1
    noise_1 = 0.1
    lower_bound_1 = -1
    upper_bound_1 = 2
    stretch_factor_1 = 1
    y_shift_1 = 0  # No shift initially
    # Generate first dataset
    X1, y1 = DS001_Sinusoidal(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1, y_shift_1)

    # Define parameters for the second dataset (After Drift)
    n_samples_2 = 300
    noise_2 = 0.1  # Keep the same noise level
    lower_bound_2 = 1  # Shifted range
    upper_bound_2 = 4
    stretch_factor_2 = 1  # Keep same frequency
    y_shift_2 = 1.5  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X2, y2 = DS001_Sinusoidal(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2, y_shift_2)

    # Define parameters for the second dataset (After Drift)
    n_samples_3 = 300
    noise_3 = 0.1  # Keep the same noise level
    lower_bound_3 = 3  # Shifted range
    upper_bound_3 = 6
    stretch_factor_3 = 1  # Keep same frequency
    y_shift_3 = 3  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X3, y3 = DS001_Sinusoidal(n_samples_3, n_features, noise_3, lower_bound_3, upper_bound_3, stretch_factor_3,
                              y_shift_3)

    # Define parameters for the second dataset (After Drift)
    n_samples_4 = 300
    noise_4 = 0.1  # Keep the same noise level
    lower_bound_4 = 5  # Shifted range
    upper_bound_4 = 8
    stretch_factor_4 = 1  # Keep same frequency
    y_shift_4 = 4.5  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X4, y4 = DS001_Sinusoidal(n_samples_4, n_features, noise_4, lower_bound_4, upper_bound_4, stretch_factor_4,
                              y_shift_4)

    # Combine both datasets
    X_drift = np.vstack((X1, X2, X3, X4))
    y_drift = np.concatenate((y1, y2, y3, y4))

    Plotter.plot_incremental_drift((X1, y1), (X2, y2), (X3, y3), (X4, y4))  # Works for any number of datasets!



