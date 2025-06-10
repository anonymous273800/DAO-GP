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
from Kernels.KernelsForTIFC import CompositeParabolicPeriodicArdRbf
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 1000
    n_features = 1
    noise_level = 0.1
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1  # This determines the period.
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound, stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 20
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    UNCERTAINTY_THRESHOLD = 0.001
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "rbf"
    MAX_INDUCING = 100  # maximum number of inducing points to retain.
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
    Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)

    #################################

    # FITC_kernel = ParabolicPeriodicLinear.ParabolicPeriodicLinear(input_dim=n_features)
    FITC_kernel = CompositeParabolicPeriodicArdRbf.CompositeParabolicPeriodicARDRBF(input_dim=n_features)
    # NOTE: WHEN YOU CHANGE THE INITIAL_BATCH_SIZE => YOU GET BETTER PERFORMANCE.
    online_gp, epoch_list_fitc, r2_list_tr_fitc = FITC.OnlineFITCGPMe(X_train, y_train, FITC_kernel, MAX_INDUCING,
                                                                      INITIAL_BATCH_SIZE, INCREMENT_SIZE, noise_level)
    y_pred, pred_var = online_gp.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    print(f"\nFinal Performance: MSE5={final_mse:.4f}, R²={final_r2:.4f}")

    x_axis_fitc_gp = epoch_list_fitc
    y_axis_fitc_gp = r2_list_tr_fitc
    label_fitc_gp = 'FITC-GP'

    # Plotter
    # Plotter.plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp)
    # Plotter.plot_minibatches_results_standard(x_axis_fitc_gp, y_axis_fitc_gp, KPI, label_fitc_gp)

    Plotter.plot_minibatches_results_standard_all(x_axis_dao_gp, y_axis_dao_gp, KPI, label_dao_gp,
                                                  x_axis_fitc_gp, y_axis_fitc_gp, label_fitc_gp, legend_loc='lower right')
