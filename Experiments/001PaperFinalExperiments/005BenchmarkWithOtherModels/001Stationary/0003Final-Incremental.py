import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from Plotter import Plotter
from Utils import Util
from sklearn.utils import shuffle
from Kernels.KernelsFunctions.K009ParabolicPlusArd import K009ParabolicPlusArd
from Models.OnlineKernalizedPassiveAggressiveRegressor import KPA
from Models.OnlineFITCGP import OnlineFITCGPKernels
from Models.OnlineFITCGP import OnlineFITCGPWithOptimization
from Datasets.Drift.Incremental.DS001Sin import DS001Sin
from Utils import DriftUtil
import warnings
warnings.filterwarnings("ignore")


def create_get_DS():
    # Define parameters for the first dataset (Before Drift)
    n_samples_1 = 500
    n_features = 3
    noise_1 = 0.1
    lower_bound_1 = -1
    upper_bound_1 = 2
    stretch_factor_1 = 1
    y_shift_1 = 0  # No shift initially
    # Generate first dataset
    X1, y1 = DS001Sin.DS001_Sinusoidal(n_samples_1, n_features, noise_1, lower_bound_1, upper_bound_1, stretch_factor_1,y_shift_1)
    X1, y1 = shuffle(X1, y1, random_state=seed)

    # Define parameters for the second dataset (After Drift)
    n_samples_2 = 500
    noise_2 = 0.1  # Keep the same noise level
    lower_bound_2 = 1  # Shifted range
    upper_bound_2 = 4
    stretch_factor_2 = 1  # Keep same frequency
    y_shift_2 = 1.5  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X2, y2 = DS001Sin.DS001_Sinusoidal(n_samples_2, n_features, noise_2, lower_bound_2, upper_bound_2, stretch_factor_2, y_shift_2)
    X2, y2 = shuffle(X2, y2, random_state=seed)
    DriftUtil.quantify_drift(X1, y1, X2, y2)

    # Define parameters for the second dataset (After Drift)
    n_samples_3 = 500
    noise_3 = 0.1  # Keep the same noise level
    lower_bound_3 = 3  # Shifted range
    upper_bound_3 = 6
    stretch_factor_3 = 1  # Keep same frequency
    y_shift_3 = 3  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X3, y3 = DS001Sin.DS001_Sinusoidal(n_samples_3, n_features, noise_3, lower_bound_3, upper_bound_3, stretch_factor_3,y_shift_3)
    X3, y3 = shuffle(X3, y3, random_state=seed)
    DriftUtil.quantify_drift(X2, y2, X3, y3)

    # Define parameters for the second dataset (After Drift)
    n_samples_4 = 500
    noise_4 = 0.1  # Keep the same noise level
    lower_bound_4 = 5  # Shifted range
    upper_bound_4 = 8
    stretch_factor_4 = 1  # Keep same frequency
    y_shift_4 = 4.5  # **Introduce y-axis shift**
    # Generate second dataset (same X range, but different y)
    X4, y4 = DS001Sin.DS001_Sinusoidal(n_samples_4, n_features, noise_4, lower_bound_4, upper_bound_4, stretch_factor_4,
                              y_shift_4)
    X4, y4 = shuffle(X4, y4, random_state=seed)
    DriftUtil.quantify_drift(X3, y3, X4, y4)

    # Combine both datasets
    X_drift = np.vstack((X1, X2, X3, X4))
    y_drift = np.concatenate((y1, y2, y3, y4))

    Plotter.plot_incremental_drift((X1, y1), (X2, y2), (X3, y3), (X4, y4), dataset_type="Inc. Drift Sin")  # Works for any number of datasets!

    return X_drift, y_drift

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    STRETCH_FACTOR = 1

    X, y = create_get_DS()
    Plotter.plot_basic(X, y)
    X, y = Util.normalize_dataset(X, y)



    # Always take the last 500 data points
    last_500_X = X[-500:]
    last_500_y = y[-500:]

    # Use 20% of those as test (i.e. 100 points)
    test_size = int(0.2 * 500)  # = 100
    X_test = last_500_X[-test_size:]
    y_test = last_500_y[-test_size:]

    # The rest (first len(X)-100 points) is training
    X_train = X[:-test_size]
    y_train = y[:-test_size]


    n_samples, n_features = X_train.shape
    # print("################# Start of DAO-GP #################")
    # # # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 50
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    MAX_INDUCING = 100
    INITIAL_KERNEL = "rbf"
    UNCERTAINTY_THRESHOLD = 0.001
    KPI = 'R2'
    Z = 3.5
    SAFE_AREA_THRESHOLD = .005
    KERNEL_POOL = KernelsPool.kernels_list

    X_base_tr, y_base_tr, K_inv, kernel, kernel_func, kernel_args, noise, epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl = \
        DAOGP.dao_gp2(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL,
                     KPI, Z, SAFE_AREA_THRESHOLD, KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR)

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

    dao_dp_final_mse = mean_squared_error(y_test, mu_test)
    dao_dp_final_r2 = r2_score(y_test, mu_test)
    print(f"\nDAO-DP - Final Test MSE: {dao_dp_final_mse:.4f}, Final Test R^2: {dao_dp_final_r2:.4f}")

    # Plot the predicted values on each mini-batch (training and validation)
    x_axis_dao_gp = epoch_list
    y_axis_dao_gp = r2_list_tr
    label_dao_gp = 'DAO-GP'
    # Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)

    print("################# END of DAO-GP #################")
    print("*************************************************************************************************")
    print("################# Start of FITC #################")
    # Online-FITC-GP
    import math
    # gamma_poly= c_poly= sigma_per= length_per = period = 1.0
    # fitc_kernel = OnlineFITCGPKernels.ParabolicPeriodicLinear(input_dim=n_features, gamma_poly=gamma_poly, c_poly=c_poly, sigma_per=sigma_per, length_per=length_per, period=period)
    gamma_poly = c_poly = sigma_per = length_per = period = 1.0
    fitc_kernel = OnlineFITCGPKernels.CompositeParabolicPeriodicARDRBF(input_dim=n_features,
                                                                       gamma_poly=gamma_poly, c_poly=c_poly,
                                                                       sigma_per=sigma_per, length_per=length_per,
                                                                       period=period)
    fitc_online_gp, fitc_r2_list, fitc_mse_list, fitc_epoch_list = OnlineFITCGPWithOptimization.OnlineFITCGPCaller(X_train, y_train, fitc_kernel, MAX_INDUCING,
                                                              INITIAL_BATCH_SIZE, INCREMENT_SIZE, noise_var=noise, optimize_every=20)

    fitc_y_pred, fitc_pred_var = fitc_online_gp.predict(X_test)
    fitc_final_mse = mean_squared_error(y_test, fitc_y_pred)
    fitc_final_r2 = r2_score(y_test, fitc_y_pred)
    print(f"\nFinal Performance: MSE={fitc_final_mse:.4f}, R²={fitc_final_r2:.4f}")

    x_axis_fitc = fitc_epoch_list
    y_axis_fitc = fitc_r2_list
    label_fitc = 'FITC-GP'
    print("################# End of FITC #################")
    print("*************************************************************************************************")
    print("################# Start of KPA #################")

    # KPA
    # Expr 0002Final-Abrupt.py
    kpa_kernel_params = {
        "gamma_poly": 0.001000,
        "c_poly": 75.065090,
        "sigma_per": 0.008209,
        "length_per": 0.629467,
        "period": 1.279281,
        "rbf_lengths": np.ones(n_features - 1),
        "rbf_sigma": 1.157831,
    }

    def kpa_kernel(X1, X2):
        return K009ParabolicPlusArd.composite_parabolic_periodic_ard_rbf(X1, X2, **kpa_kernel_params)


    # kpa_kernel_params = {
    #     "per_length": 0.526628,
    #     "per_sigma": 1.250962,
    #     "period": 1.034145,
    #     "beta_lin": 0.001000,
    # }

    # def kpa_kernel(X1, X2):
    #     return K010PeriodicPlusLinear.composite_periodic_plus_linear(X1, X2, **kpa_kernel_params)

    kpa_model, kpa_r2_list, kpa_mse_list, kpa_epoch_list =  KPA.kpa(X_train, y_train, kpa_kernel,INCREMENT_SIZE, C=1.0, epsilon=0.1)
    # Final evaluation
    kpa_y_pred = kpa_model.predict(X_test)
    kpa_r2 = r2_score(y_test, kpa_y_pred)
    kpa_mse = mean_squared_error(y_test, kpa_y_pred)
    print(f"\nFinal Test R² = {kpa_r2:.4f}")
    print(f"Final Test MSE = {kpa_mse:.4f}")

    x_axis_kpa = kpa_epoch_list
    y_axis_kpa = kpa_r2_list
    label_kpa = 'OKPA'
    print("################# End of KPA #################")
    print("*************************************************************************************************")
    print("################ Summary Final Results ################")
    print("DAO-GP Test-Data", "MSE", f"{dao_dp_final_mse:.4f}", "R²", f"{dao_dp_final_r2:.4f}")
    print("FTIC Test-Data", "MSE", f"{fitc_final_mse:.4f}", "R²", f"{fitc_final_r2:.4f}")
    print("KPA Test-Data", "MSE", f"{kpa_mse:.4f}", "R²", f"{kpa_r2:.4f}")
    print("*************************************************************************************************")

    ################# Plotting #########################
    Plotter.plot_minibatches_results_standard_all_incremental(KPI, x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                  x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                  x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                  X_test, y_test,
                                                  legend_loc='lower left', drift_locations_every=500)