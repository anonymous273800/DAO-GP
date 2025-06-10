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

from Models.FITC import FITC
from Kernels.KernelsForTIFC import ParabolicPeriodicLinear


from Models.KRLS import KRLS


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 2000
    n_features = 1
    noise_level = 0.01
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1  # This determines the period.
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound, stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 50
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    UNCERTAINTY_THRESHOLD = 0.001
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "composite_parabolic_periodic_ard_rbf"
    MAX_INDUCING = 100
    KPI = 'R2'
    Z = 3.5
    SAFE_AREA_THRESHOLD = .005
    KERNEL_POOL = KernelsPool.kernels_list

    X_base_tr, y_base_tr, K_inv, kernel, kernel_func, kernel_args, noise, epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl = \
        DAOGP.dao_gp(X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA, MAX_INDUCING, INITIAL_KERNEL,
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
    # Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)

    ###################################
    FITC_kernel = ParabolicPeriodicLinear.ParabolicPeriodicLinear(input_dim=n_features)

    online_gp, epoch_list_fitc, r2_list_tr_fitc = FITC.OnlineFITCGPMe(X_train, y_train, FITC_kernel, MAX_INDUCING, INITIAL_BATCH_SIZE, INCREMENT_SIZE,noise_level)
    y_pred, pred_var = online_gp.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    print(f"\nFinal Performance: MSE5={final_mse:.4f}, R²={final_r2:.4f}")

    x_axis_fitc_gp = epoch_list_fitc
    y_axis_fitc_gp = r2_list_tr_fitc
    label_fitc_gp = 'FITC-GP'

    #####################################
    # KRLS

    # --- Simple RBF Kernel ---
    def rbf_kernel(X1, X2, sigma):
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-dists / (2 * sigma ** 2))

    krls_lambdaa = 1e-6  # low ALD threshold to allow dictionary growth
    krls_sigma = 0.3
    krls_model, krls_r2_list, krls_mse_list, krls_epoch_list = KRLS.kernelized_RLS(X_train, y_train, krls_lambdaa, krls_sigma, rbf_kernel)

    # Predict
    krls_y_pred = np.array([krls_model.predict(xi) for xi in X_test])

    # Evaluate
    print("R²:", r2_score(y_test, krls_y_pred))
    print("MSE:", mean_squared_error(y_test, krls_y_pred))
    print("Final dictionary size:", len(krls_model.dictionary))

    x_axis_krls_gp = krls_epoch_list
    y_axis_krls_gp = krls_r2_list
    label_krls_gp = 'KRLS'

    ### Plotting ###
    Plotter.plot_minibatches_results_standard_all(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp,
                                                  x_axis_fitc_gp, y_axis_fitc_gp, label_fitc_gp,
                                                  x_axis_krls_gp, y_axis_krls_gp, label_krls_gp,
                                                  legend_loc='lower right')



