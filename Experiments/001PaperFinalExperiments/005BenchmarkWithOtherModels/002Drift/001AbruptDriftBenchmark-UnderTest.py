import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from Plotter import Plotter
from Utils import Util
import warnings
from Models.OnlineFITCGP import OnlineFITCGP
from Models.KRLS import KRLS
from Models.OnlineKernelRidgeRegression import OnlineKernelRidgeRegression
from Kernels.KernelsFunctions.K009ParabolicPlusArd import K009ParabolicPlusArd
from Kernels.KernelsFunctions.K001RBF import K001RBF
from Models.OnlineKernalizedPassiveAggressiveRegressor import KPA
from Models.OnlineFITCGP import OnlineFITCGPKernels
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from Datasets.Standard.Synthetic.DS007ParabolicWave import DS007ParabolicWave
from Utils import DriftUtil
from Datasets.Standard.Synthetic.DS007ParabolicWaveAdvanced import DS007ParabolicWaveAdvanced
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 1000
    n_features = 3
    noise_level = 0.1
    lower_bound = -10
    upper_bound = 10
    stretch_factor = 4
    additional_features_weight = 0.3
    X1, y1 = DS007ParabolicWaveAdvanced.DS007_ParabolicWave_Simple(n_samples, n_features, noise_level,
        lower_bound, upper_bound, stretch_factor, additional_features_weight)
    X1, y1 = shuffle(X1, y1, random_state=seed)

    n_samples_2 = 1000
    n_features_2 = 3
    noise_level_2 = 0.1
    lower_bound_2 = 0
    upper_bound_2 = 20
    stretch_factor_2 = 8
    additional_features_weight_2 = 0.8
    X2, y2 = DS007ParabolicWaveAdvanced.DS007_ParabolicWave_Simple(n_samples_2, n_features_2, noise_level_2,
        lower_bound_2, upper_bound_2, stretch_factor_2, additional_features_weight_2)
    X2, y2 = shuffle(X2, y2, random_state=seed)
    DriftUtil.quantify_drift(X1, y1, X2, y2)

    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    # X, y = Util.normalize_dataset(X, y)

    test_fraction = 0.2
    split_index = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]


    # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 50
    INCREMENT_SIZE = 10
    DECAY_GAMMA = .99
    UNCERTAINTY_THRESHOLD = 0.001

    INITIAL_KERNEL = "rbf"
    MAX_INDUCING = 50  # maximum number of inducing points to retain.
    KPI = 'R2'
    Z = 2.5
    SAFE_AREA_THRESHOLD = .005
    KERNEL_POOL = KernelsPool.kernels_list
    STRETCH_FACTOR = 1

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

    final_mse = mean_squared_error(y_test, mu_test)
    final_r2 = r2_score(y_test, mu_test)
    print(f"\nFinal Test MSE: {final_mse:.4f}, Final Test R^2: {final_r2:.4f}")

    Plotter.plot_final(
        X_base_tr, y_base_tr,
        K_inv_undecayed=K_inv,
        gamma=DECAY_GAMMA,
        dataset_type="sinusoidal",
        X_test=X_test,
        y_test=y_test,
        noise=noise,
        kernel_func=kernel_func, final_mse=final_mse, final_r2=final_r2,
        **kernel_args
    )

    # Plot the predicted values on each mini-batch (training and validation)
    dao_dp_final_mse = mean_squared_error(y_test, mu_test)
    dao_dp_final_r2 = r2_score(y_test, mu_test)
    print(f"\nDAO-DP - Final Test MSE: {dao_dp_final_mse:.4f}, Final Test R^2: {dao_dp_final_r2:.4f}")

    # Plot the predicted values on each mini-batch (training and validation)
    x_axis_dao_gp = epoch_list
    y_axis_dao_gp = r2_list_tr
    label_dao_gp = 'DAO-GP'
    # print("################# END of DAO-GP #################")
    print("*************************************************************************************************")
    print("################# Start of FITC #################")
    # Online-FITC-GP
    # fitc_kernel = OnlineFITCGPKernels.ParabolicPeriodicLinear(input_dim=n_features)
    fitc_kernel = OnlineFITCGPKernels.CompositeParabolicPeriodicARDRBF(input_dim=n_features)
    fitc_online_gp, fitc_r2_list, fitc_mse_list, fitc_epoch_list = OnlineFITCGP.OnlineFITCGPCaller(X_train, y_train,
                                                                                                   fitc_kernel,
                                                                                                   MAX_INDUCING,
                                                                                                   INITIAL_BATCH_SIZE,
                                                                                                   INCREMENT_SIZE)

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
    kpa_kernel_params = {
        "gamma_poly": 100,
        "c_poly": 0.001,
        "sigma_per": .014,
        "length_per": .58,
        "period": 6.28,
        "rbf_lengths": np.ones(n_features - 1),
        "rbf_sigma": 1.0,
    }


    def kpa_kernel(X1, X2):
        # return K001RBF.rbf_kernel_ard(X1, X2, 1,1)
        return K009ParabolicPlusArd.composite_parabolic_periodic_ard_rbf(X1, X2, **kpa_kernel_params)


    kpa_model, kpa_r2_list, kpa_mse_list, kpa_epoch_list = KPA.kpa(X_train, y_train, kpa_kernel, INCREMENT_SIZE, C=1, epsilon=.01)
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
    # print("*************************************************************************************************")
    #
    ################# Plotting #########################
    Plotter.plot_minibatches_results_standard_all2(KPI, x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                   x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                   x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                   X_test, y_test,
                                                   legend_loc='lower left')



