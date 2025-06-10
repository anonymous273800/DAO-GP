import numpy as np

import Kernels.KernelsPoolManager.KernelsPool
from Datasets.Drift.Gradual.DS001Sin import DS001Sin
from sklearn.metrics import mean_squared_error, r2_score
from Utils import Util
from Plotter import Plotter
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")


def create_get_DS():
    # Define parameters for the first dataset (Before Drift)
    n_samples_1 = 2500
    n_features = 50
    noise_1 = 0.2
    lower_bound_1 = -1
    upper_bound_1 = 2
    stretch_factor_1 = 1
    y_shift_1 = 0  # No shift initially
    # Generate first dataset
    X1, y1 = DS001Sin.DS001_Sinusoidal(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1,
                              y_shift_1)
    X1, y1 = shuffle(X1, y1, random_state=seed)

    # Define parameters for the second dataset (After Drift)
    n_samples_2 = 500
    noise_2 = 0.2  # Keep the same noise level
    lower_bound_2 = 5  # Shifted range
    upper_bound_2 = 8
    stretch_factor_2 = 1  # Keep same frequency
    y_shift_2 = 3  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X2, y2 = DS001Sin.DS001_Sinusoidal(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2,
                              y_shift_2)
    X2, y2 = shuffle(X2, y2, random_state=seed)

    # Define parameters for the first dataset (Before Drift)
    n_samples_3 = 500

    noise_3 = 0.2
    lower_bound_3 = -1
    upper_bound_3 = 2
    stretch_factor_3 = 1
    y_shift_3 = 0  # No shift initially
    # Generate first dataset
    X3, y3 = DS001Sin.DS001_Sinusoidal(n_samples_3, n_features, noise_3, lower_bound_3, upper_bound_3, stretch_factor_3,
                              y_shift_3)
    X3, y3 = shuffle(X3, y3, random_state=seed)

    # Define parameters for the second dataset (After Drift)
    n_samples_4 = 500
    noise_4 = 0.2  # Keep the same noise level
    lower_bound_4 = 5  # Shifted range
    upper_bound_4 = 8
    stretch_factor_4 = 1  # Keep same frequency
    y_shift_4 = 3  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X4, y4 = DS001Sin.DS001_Sinusoidal(n_samples_4, n_features, noise_4, lower_bound_4, upper_bound_4, stretch_factor_4,
                              y_shift_4)
    X4, y4 = shuffle(X4, y4, random_state=seed)

    # Define parameters for the first dataset (Before Drift)
    n_samples_5 = 500

    noise_5 = 0.2
    lower_bound_5 = -1
    upper_bound_5 = 2
    stretch_factor_5 = 1
    y_shift_5 = 0  # No shift initially
    # Generate first dataset
    X5, y5 = DS001Sin.DS001_Sinusoidal(n_samples_5, n_features, noise_5, lower_bound_5, upper_bound_5, stretch_factor_5,
                              y_shift_5)
    X5, y5 = shuffle(X5, y5, random_state=seed)

    # Define parameters for the second dataset (After Drift)
    n_samples_6 = 2500
    noise_6 = 0.2  # Keep the same noise level
    lower_bound_6 = 5  # Shifted range
    upper_bound_6 = 8
    stretch_factor_6 = 1  # Keep same frequency
    y_shift_6 = 3  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X6, y6 = DS001Sin.DS001_Sinusoidal(n_samples_6, n_features, noise_6, lower_bound_6, upper_bound_6, stretch_factor_6,
                              y_shift_6)
    X6, y6 = shuffle(X6, y6, random_state=seed)

    # Combine both datasets
    X_drift = np.vstack((X1, X2, X3, X4, X5, X6))
    y_drift = np.concatenate((y1, y2, y3, y4, y5, y6))

    Plotter.plot_incremental_drift((X1, y1), (X2, y2), (X3, y3), (X4, y4), (X5, y5),
                                   (X6, y6))  # Works for any number of datasets!
    return X_drift, y_drift

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    STRETCH_FACTOR = 1

    X, y = create_get_DS()
    Plotter.plot_basic(X,y)
    X,y = Util.normalize_dataset(X,y)

    INITIAL_BATCH_SIZE = 200
    INCREMENT_SIZE = 25
    DECAY_GAMMA = .99
    MAX_INDUCING = 100
    INITIAL_KERNEL = "rbf"
    UNCERTAINTY_THRESHOLD = 0.001

    # Always take the last 500 data points
    last_2500_X = X[-2500:]
    last_2500_y = y[-2500:]

    # Use 20% of those as test (i.e. 100 points)
    test_size = int(0.2 * 2500)  # = 100
    X_test = last_2500_X[-test_size:]
    y_test = last_2500_y[-test_size:]

    # The rest (first len(X)-100 points) is training
    X_train = X[:-test_size]
    y_train = y[:-test_size]

    KPI = 'R2'
    Z = 2.5
    SAFE_AREA_THRESHOLD = .005

    KERNEL_POOL = KernelsPool.kernels_list

    X_base_tr, y_base_tr, K_inv, kernel, kernel_func, kernel_args, noise,\
        epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl= \
        DAOGP.dao_gp2(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL, KPI, Z, SAFE_AREA_THRESHOLD, KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR)

    # Predict on test set
    mu_test, _ = Util.computeGP(
        X_window=X_base_tr,
        y_window=y_base_tr,
        K_inv=K_inv,
        X_star=X_test,
        noise=noise,
        kernel_func=kernel_func,
        **kernel_args
    )

    final_mse = mean_squared_error(y_test, mu_test)
    final_r2 = r2_score(y_test, mu_test)
    print(f"\nFinal Test MSE: {final_mse:.4f}, Final Test R^2: {final_r2:.4f}")


    Plotter.plot_final(
        X_base_tr, y_base_tr,
        K_inv_undecayed=K_inv,
        gamma=DECAY_GAMMA,
        X_test=X_test,
        y_test=y_test,
        noise=noise,
        kernel_func=kernel_func, final_mse=final_mse, final_r2=final_r2,
        **kernel_args
    )

