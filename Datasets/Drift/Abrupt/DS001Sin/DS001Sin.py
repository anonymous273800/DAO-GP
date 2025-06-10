import numpy as np
import matplotlib.pyplot as plt
from Plotter import Plotter
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from Utils import DriftUtil

def DS001_Sinusoidal_Not_Normalized(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1, y_shift=0.0):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))  # Random values for multi-dim X
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + noise * np.random.randn(n_samples)  + y_shift
    # y = y.ravel()
    return X, y

def DS001_Sinusoidal_Normalized(n_samples, n_features, noise=0.1, lower_bound=0, upper_bound=10, stretch_factor=1, y_shift=0):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + noise * np.random.randn(n_samples)

    # Normalize X and y
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel() + y_shift

    return X, y

def DS001_Drift_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1, y_shift_1,
                           n_samples_2, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2, y_shift_2, seed):

    # Generate first dataset
    X1, y1 = DS001_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1,
                              y_shift_1)
    X1, y1 = shuffle(X1, y1, random_state=seed)


    # Generate second dataset (same X range, but different y)
    X2, y2 = DS001_Sinusoidal_Not_Normalized(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2, y_shift_2)
    X2, y2 = shuffle(X2, y2, random_state=seed)

    # DriftUtil.quantify_drift(X1, y1, X2, y2)
    DriftUtil.quantify_drift(X1, y1, X2, y2)

    # Combine both datasets
    X_drift = np.vstack((X1, X2))
    y_drift = np.concatenate((y1, y2))
    return X_drift, y_drift


if __name__ == "__main__":

    # # Define parameters for the first dataset (Before Drift)
    # n_samples_1 = 300
    # n_features = 1
    # noise_1 = 0.1
    # lower_bound_1 = -1
    # upper_bound_1 = 2
    # stretch_factor_1 = 1
    # y_shift_1 = 0  # No shift initially
    # # Generate first dataset
    # X1, y1 = DS001_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1, y_shift_1)
    #
    # # Define parameters for the second dataset (After Drift)
    # n_samples_2 = 300
    # noise_2 = 0.2  # Keep the same noise level
    # lower_bound_2 = 5  # Shifted range
    # upper_bound_2 = 8
    # stretch_factor_2 = 1  # Keep same frequency
    # y_shift_2 = 4.5  # **Introduce y-axis shift**
    # # Generate second dataset (same X range, but different y)
    # X2, y2 = DS001_Sinusoidal_Not_Normalized(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2, y_shift_2)
    #
    # # Combine both datasets
    # X_drift = np.vstack((X1, X2))
    # y_drift = np.concatenate((y1, y2))
    #
    # Plotter.plot_abrupt_drift(X1, y1, X2, y2)

    ##################
    # n_samples_2 = 1000
    # n_features = 1
    # noise_2 = 0.1
    # lower_bound_2 = 4
    # upper_bound_2 = 8
    # y_shift_2 = 10
    #
    # X, y = DS001_Sinusoidal_Not_Normalized(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor=1, y_shift=y_shift_2)
    # Plotter.plot_basic(X,y)

    ###############

    # # Abrupt DS001Sin
    # n_samples_1 = 1000
    # n_features = 1
    # noise_1 = 0.1
    # lower_bound_1 = -1
    # upper_bound_1 = 2
    # stretch_factor_1 = 1
    # y_shift_1 = 0.0  # No shift initially
    # n_samples_2 = 1000
    # noise_2 = 0.1  # Keep the same noise level
    # lower_bound_2 = 5  # Shifted range
    # upper_bound_2 = 8
    # stretch_factor_2 = 1  # Keep same frequency
    # y_shift_2 = 4.5  # **Introduce y-axis shift**
    #
    # # Generate first dataset
    # X1, y1 = DS001_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1,
    #                                          stretch_factor_1,
    #                                          y_shift_1)
    # X1, y1 = shuffle(X1, y1, random_state=42)
    #
    # X2, y2 = DS001_Sinusoidal_Not_Normalized(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2,
    #                                          stretch_factor_2, y_shift_2)
    # X2, y2 = shuffle(X2, y2, random_state=42)
    #
    # # Combine both datasets
    # X_drift = np.vstack((X1, X2))
    # y_drift = np.concatenate((y1, y2))
    #
    # # X, y = DS001_Drift_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1,
    # #                                                       upper_bound_1,
    # #                                                       stretch_factor_1, y_shift_1,
    # #                                                       n_samples_2, noise_2, lower_bound_2, upper_bound_2,
    # #                                                       stretch_factor_2,
    # #                                                       y_shift_2,
    # #                                                       seed=42)
    # Plotter.plot_abrupt_drift(X1, y1, X2, y2, dataset_type='Abrupt Drift Sin')

    seed = 42
    np.random.seed(seed)
    STRETCH_FACTOR = 1

    # Abrupt DS001Sin
    n_samples_1 = 1500
    n_features = 3
    noise_1 = 0.1
    lower_bound_1 = -1
    upper_bound_1 = 2
    stretch_factor_1 = 1
    y_shift_1 = 0  # No shift initially
    n_samples_2 = 1500
    noise_2 = 0.1  # Keep the same noise level
    lower_bound_2 = 5  # Shifted range
    upper_bound_2 = 8
    stretch_factor_2 = 1  # Keep same frequency
    y_shift_2 = 4.5  # **Introduce y-axis shift**
    X, y = DS001_Drift_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1,
                                                          upper_bound_1,
                                                          stretch_factor_1, y_shift_1,
                                                          n_samples_2, noise_2, lower_bound_2, upper_bound_2,
                                                          stretch_factor_2,
                                                          y_shift_2,
                                                          seed)

    # seed = 42
    # np.random.seed(seed)
    # STRETCH_FACTOR = 1
    #
    # # Abrupt DS001Sin
    # n_samples_1 = 5000
    # n_features = 50
    # noise_1 = 0.2
    # lower_bound_1 = -1
    # upper_bound_1 = 2
    # stretch_factor_1 = 1
    # y_shift_1 = 0  # No shift initially
    # n_samples_2 = 5000
    # noise_2 = 0.2  # Keep the same noise level
    # lower_bound_2 = 5  # Shifted range
    # upper_bound_2 = 8
    # stretch_factor_2 = 1  # Keep same frequency
    # y_shift_2 = 4.5  # **Introduce y-axis shift**
    # X, y = DS001_Drift_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1,
    #                                                       upper_bound_1,
    #                                                       stretch_factor_1, y_shift_1,
    #                                                       n_samples_2, noise_2, lower_bound_2, upper_bound_2,
    #                                                       stretch_factor_2,
    #                                                       y_shift_2,
    #                                                       seed)

