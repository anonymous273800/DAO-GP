import numpy as np

import Kernels.KernelsPoolManager.KernelsPool
from Datasets.Drift.Abrupt.DS001Sin import DS001Sin
from sklearn.metrics import mean_squared_error, r2_score
from Utils import Util
from Plotter import Plotter
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
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
    X, y = DS001Sin.DS001_Drift_Sinusoidal_Not_Normalized(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1,
                                           stretch_factor_1, y_shift_1,
                                           n_samples_2, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2,
                                           y_shift_2,
                                           seed)

    Plotter.plot_basic(X,y)
    X,y = Util.normalize_dataset(X,y)

    INITIAL_BATCH_SIZE = 50
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    MAX_INDUCING = 100
    INITIAL_KERNEL = "rbf"
    UNCERTAINTY_THRESHOLD = 0.001

    test_fraction = 0.2
    split_index = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    KPI = 'R2'
    Z = 2.5
    SAFE_AREA_THRESHOLD = .005

    KERNEL_POOL = KernelsPool.kernels_list

    X_base_tr, y_base_tr, K_inv, kernel, kernel_func, kernel_args, noise,\
        epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl= \
        DAOGP.dao_gp(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL, KPI, Z, SAFE_AREA_THRESHOLD, KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR)

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

    # Plot the predicted values on each mini-batch (training and validation)
    x_axis_dao_gp = epoch_list
    y_axis_dao_gp = r2_list_tr
    label_dao_gp = 'DAO-GP'
    Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)

