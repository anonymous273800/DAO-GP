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
import matplotlib.pyplot as plt
from Datasets.Standard.Public.DS100AppaRealFace import RealFaceDS
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    # zip_path = 'C:/PythonProjects/OnlineGPNonlinear/Datasets/Standard/Public/DS100AppaRealFace/DSAppaRealFace.zip'
    # X, y = RealFaceDS.get_appa_real_face_ds(zip_path, img_size=(32, 32), max_images=400)
    zip_path = "C:/PythonProjects/OnlineGPNonLinear/Datasets/Standard/Public/DS100AppaRealFace/DSAppaRealFace.zip"
    extract_path = "C:/PythonProjects/OnlineGPNonLinear/Datasets/Standard/Public/DS100AppaRealFace/appa_extracted"

    # X, y = RealFaceDS.get_appa_realface_ds(zip_path, extract_path)
    X, y = RealFaceDS.get_appa_realface_ds_with_pca(zip_path, extract_path, image_size=(64, 64), use_grayscale=True)

    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 500
    INCREMENT_SIZE = 500
    DECAY_GAMMA = 1
    UNCERTAINTY_THRESHOLD = 0.001
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "rbf"
    MAX_INDUCING = 1000  # maximum number of inducing points to retain.
    KPI = 'R2'
    Z = 2.5
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
