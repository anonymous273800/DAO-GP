import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import warnings

# Your project-specific imports (assumed available in your local environment)
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from Utils import Util

warnings.filterwarnings("ignore")

# --- Kernel Parameter Class ---
class ParabolicPlusLinear:
    @staticmethod
    def get_params(stretch_factor):
        return [
            {"name": "gamma_poly", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "c_poly", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "sigma_per", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "length_per", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "period", "init": 2 * math.pi / stretch_factor, "bounds": (2 * math.pi / stretch_factor, 2 * math.pi / stretch_factor)},
            {"name": "beta_lin", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "noise", "init": 0.1, "bounds": (0.1, 10)}
        ]

def get_kernel_args_from_param_list(param_list):
    return {p["name"]: p["init"] for p in param_list if p["name"] != "noise"}

# --- Kernel Function ---
def parabolic_linear_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period, beta_lin):
    x1 = X1[:, 0:1]
    x2 = X2[:, 0:1]
    K_nonlin = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    diff = np.abs(x1 - x2.T)
    K_per = sigma_per ** 2 * np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))
    K1 = K_nonlin * K_per

    if X1.shape[1] > 1 and X2.shape[1] > 1:
        K_lin = beta_lin ** 2 * (X1[:, 1:] @ X2[:, 1:].T)
    else:
        K_lin = 0

    return K1 + K_lin

# --- PA Model ---
class KernelizedPARegressor:
    def __init__(self, C=0.01, epsilon=0.1, kernel_func=None, kernel_args={}):
        self.C = C
        self.epsilon = epsilon
        self.kernel_func = kernel_func
        self.kernel_args = kernel_args
        self.support_vectors = []
        self.alpha = []

    def predict(self, X):
        if not self.support_vectors:
            return np.zeros(X.shape[0])
        K = self.kernel_func(np.array(self.support_vectors), X, **self.kernel_args)
        return np.dot(self.alpha, K)

    def partial_fit(self, X, y):
        for i in range(X.shape[0]):
            xi = X[i:i + 1]
            yi = y[i]
            y_pred = self.predict(xi)[0]
            loss = max(0, abs(yi - y_pred) - self.epsilon)
            if loss > 0:
                k_xi_xi = self.kernel_func(xi, xi, **self.kernel_args)[0, 0]
                tau = loss / (k_xi_xi + 1 / (2 * self.C))
                self.alpha.append(np.sign(yi - y_pred) * tau)
                self.support_vectors.append(xi[0])

# --- Train PA ---
def non_linear_PA(X_train, y_train, stretch_factor=1):
    param_list = ParabolicPlusLinear.get_params(stretch_factor)
    kernel_args = get_kernel_args_from_param_list(param_list)
    model = KernelizedPARegressor(
        C=0.01, epsilon=0.1,
        kernel_func=parabolic_linear_kernel,
        kernel_args=kernel_args
    )
    model.partial_fit(X_train, y_train)
    return model

# --- DAO-GP Wrapper ---
def DAO_GP(X_train, y_train):
    return DAOGP.dao_gp(
        X_train, y_train,
        20, 20, .99,
        100,
        'rbf', 'R2',
        2.5, .005,
        KernelsPool.kernels_list,
        0.001, 1
    )

# --- MAIN ---
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    # Dataset
    X, y = DS001Sin.DS001_Sinusoidal(
        1000, 2, 0.1, -5, 5, stretch_factor=1
    )
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # --- DAO-GP Evaluation ---
    print("Running DAO-GP…")
    Xb, yb, Kinv, kern, kern_func, kern_args, noise, *_ = DAO_GP(X_train, y_train)
    mu, _ = Util.computeGP(
        X_window=Xb, y_window=yb, K_inv=Kinv,
        X_star=X_test, noise=noise,
        kernel_func=kern_func, **kern_args
    )
    print(
        f"DAO-GP Final Test MSE: {mean_squared_error(y_test, mu):.4f}, "
        f"Final Test R^2: {r2_score(y_test, mu):.4f}"
    )

    # --- PA Evaluation ---
    print("Running Non-linear Passive-Aggressive…")
    pa_model = non_linear_PA(X_train, y_train, stretch_factor=1)
    pa_preds = pa_model.predict(X_test)
    print(
        f"Non-linear PA Final Test MSE: {mean_squared_error(y_test, pa_preds):.4f}, "
        f"Final Test R^2: {r2_score(y_test, pa_preds):.4f}"
    )
