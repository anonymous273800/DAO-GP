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
# from Models.OnlineFITCGP import OnlineFITCGP
from Models.OnlineFITCGP import OnlineFITCGPWithOptimization
from Models.KRLS import KRLS
from Models.OnlineKernelRidgeRegression import OnlineKernelRidgeRegression
from Kernels.KernelsFunctions.K001RBF import K001RBF
from Kernels.KernelsFunctions.K002Periodic import K002Periodic
from Kernels.KernelsFunctions.K003RationalQuadratic import K003RationalQuadratic
from Kernels.KernelsFunctions.K004PeriodicPlusArd import K004PeriodicPlusArd
from Kernels.KernelsFunctions.K005Polynomial import K005Polynomial
from Kernels.KernelsFunctions.K006LogPlusLinear import K006LogPlusLinear
from Kernels.KernelsFunctions.K007ChangePoint import K007ChangePoint
from Kernels.KernelsFunctions.K008ParabolicLinear import K008ParabolicLinear
from Kernels.KernelsFunctions.K009ParabolicPlusArd import K009ParabolicPlusArd
from Models.OnlineKernalizedPassiveAggressiveRegressor import KPA
from Models.OnlineFITCGP import OnlineFITCGPKernels
from Datasets import PublicDS
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    path = Util.get_dataset_path("insurance.csv")
    X, y = PublicDS.get_medical_cost_insurance_DS(path)

    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    n_samples, n_features = X_train.shape
    # print("################# Start of DAO-GP #################")
    # # # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 100
    INCREMENT_SIZE = 10
    DECAY_GAMMA = 1
    UNCERTAINTY_THRESHOLD = 0.001
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "composite_parabolic_periodic_ard_rbf"
    MAX_INDUCING = 100
    KPI = 'R2'
    Z = 1.5
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
    Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)

    print("################# END of DAO-GP #################")
    print("*************************************************************************************************")
    print("################# Start of FITC #################")
    # Online-FITC-GP
    # fitc_kernel = OnlineFITCGPKernels.ParabolicPeriodicLinear(input_dim=n_features)
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
    # KPA
    # Expr 0002Final-Abrupt.py
    kpa_kernel_params = {
        "gamma_poly": 0.139972,
        "c_poly": 9.784419,
        "sigma_per": 0.840112,
        "length_per": 100.000000,
        "period": 6.283185,
        "rbf_lengths": np.ones(n_features - 1),
        "rbf_sigma": 0.001000,
    }


    def kpa_kernel(X1, X2):
        return K009ParabolicPlusArd.composite_parabolic_periodic_ard_rbf(X1, X2, **kpa_kernel_params)

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
    Plotter.plot_minibatches_results_standard_all2(KPI, x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                  x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                  x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                  X_test, y_test,
                                                  legend_loc='lower right')