import numpy as np
import tensorflow as tf
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from Utils import Util
from sklearn.metrics import mean_squared_error, r2_score
import gpflow
from Models.OSGPR.osgpr import OSGPR_VFE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

gpflow.config.set_default_float(tf.float32)

def osgpr(X_train, y_train, batch_size=100, num_inducing=30, seed=42):
    np.random.seed(seed)
    N = X_train.shape[0]
    assert N >= 3 * batch_size, "Need at least 3 batches of data"

    # Periodic kernel for sinusoidal data
    base_kernel = gpflow.kernels.RBF(lengthscales=1.0)
    kernel = gpflow.kernels.Periodic(base_kernel)
    kernel.period.assign(2 * np.pi)  # Set appropriate period for sine wave

    opt = gpflow.optimizers.Scipy()

    # Batch 1
    X1, y1 = X_train[:batch_size], y_train[:batch_size].reshape(-1, 1)
    Z1 = KMeans(n_clusters=num_inducing, random_state=seed).fit(X1).cluster_centers_
    model1 = gpflow.models.SGPR((X1, y1), kernel, inducing_variable=Z1, noise_variance=0.01)
    model1.inducing_variable.trainable = True
    opt.minimize(model1.training_loss, model1.trainable_variables, options={"maxiter": 500})

    mu1, Su1 = model1.predict_f(Z1, full_cov=True)
    if len(Su1.shape) == 3:
        Su1 = Su1[0]
    Kaa1 = model1.kernel(Z1)

    # Batch 2
    X2, y2 = X_train[batch_size:2*batch_size], y_train[batch_size:2*batch_size].reshape(-1, 1)
    Z2 = KMeans(n_clusters=num_inducing, random_state=seed+1).fit(X2).cluster_centers_
    model2 = OSGPR_VFE((X2, y2), kernel, mu1, Su1, Kaa1, Z1, Z2)
    model2.likelihood.variance.assign(model1.likelihood.variance)
    model2.inducing_variable.trainable = True
    opt.minimize(model2.training_loss, model2.trainable_variables, options={"maxiter": 500})

    mu2, Su2 = model2.predict_f(Z2, full_cov=True)
    if len(Su2.shape) == 3:
        Su2 = Su2[0]
    Kaa2 = model2.kernel(Z2)

    # Batch 3
    X3, y3 = X_train[2*batch_size:3*batch_size], y_train[2*batch_size:3*batch_size].reshape(-1, 1)
    Z3 = KMeans(n_clusters=num_inducing, random_state=seed+2).fit(X3).cluster_centers_
    model3 = OSGPR_VFE((X3, y3), kernel, mu2, Su2, Kaa2, Z2, Z3)
    model3.likelihood.variance.assign(model2.likelihood.variance)
    model3.inducing_variable.trainable = True
    opt.minimize(model3.training_loss, model3.trainable_variables, options={"maxiter": 500})

    return model3

def DAO_GP(X_train, y_train):
    INITIAL_BATCH_SIZE = 20
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    UNCERTAINTY_THRESHOLD = 0.001
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "rbf"
    MAX_INDUCING = 100
    KPI = 'R2'
    Z = 2.5
    SAFE_AREA_THRESHOLD = .005
    KERNEL_POOL = KernelsPool.kernels_list

    return DAOGP.dao_gp(
        X_train, y_train, INITIAL_BATCH_SIZE, INCREMENT_SIZE, DECAY_GAMMA,
        MAX_INDUCING, INITIAL_KERNEL, KPI, Z, SAFE_AREA_THRESHOLD,
        KERNEL_POOL, UNCERTAINTY_THRESHOLD, STRETCH_FACTOR)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 1000
    n_features = 2
    noise = 0.1
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1

    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # DAO-GP
    X_base_tr, y_base_tr, K_inv, kernel, kernel_func, kernel_args, noise, epoch_list, r2_list_tr, mse_list_tr, r2_list_vl, mse_list_vl = DAO_GP(X_train, y_train)
    mu_test, _ = Util.computeGP(X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv, X_star=X_test, noise=noise, kernel_func=kernel_func, **kernel_args)
    final_mse = mean_squared_error(y_test, mu_test)
    final_r2 = r2_score(y_test, mu_test)
    print(f"\nDAO-GP Final Test MSE: {final_mse:.4f}, Final Test R^2: {final_r2:.4f}")

    # OSGPR
    model_osgpr = osgpr(X_train, y_train, batch_size=100, num_inducing=100, seed=seed)
    mu_osgpr, _ = model_osgpr.predict_f(X_test)
    y_pred_osgpr = mu_osgpr.numpy().flatten()
    final_mse_osgpr = mean_squared_error(y_test, y_pred_osgpr)
    final_r2_osgpr = r2_score(y_test, y_pred_osgpr)
    print(f"OSGPR Final Test MSE: {final_mse_osgpr:.4f}, Final Test R^2: {final_r2_osgpr:.4f}")
