import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from Models.DAOGP import DAOGP
from Kernels.KernelsPoolManager import KernelsPool
from Plotter import Plotter
from Utils import Util
import GPy
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


import GPy
import numpy as np

class ParabolicPeriodicLinear(GPy.kern.Kern):
    def __init__(self, input_dim, active_dims=None, name='ParabolicPeriodicLinear'):
        super(ParabolicPeriodicLinear, self).__init__(input_dim, active_dims, name)

        # Parameters for parabolic + periodic component (on dim 0)
        self.gamma_poly = GPy.core.Param('gamma_poly', 1.0)
        self.c_poly = GPy.core.Param('c_poly', 1.0)
        self.sigma_per = GPy.core.Param('sigma_per', 1.0)
        self.length_per = GPy.core.Param('length_per', 1.0)
        self.period = GPy.core.Param('period', 1.0)

        # Parameter for linear component (dims 1+)
        self.beta_lin = GPy.core.Param('beta_lin', 1.0)

        self.link_parameters(self.gamma_poly, self.c_poly, self.sigma_per,
                             self.length_per, self.period, self.beta_lin)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # First feature only
        x1 = X[:, 0:1]
        x2 = X2[:, 0:1]
        dot_poly = self.gamma_poly * (x1 @ x2.T) + self.c_poly
        K_poly = dot_poly ** 2

        diff = np.abs(x1 - x2.T)
        K_per = self.sigma_per ** 2 * np.exp(-2 * (np.sin(np.pi * diff / self.period) ** 2) / (self.length_per ** 2))
        K_nonlin = K_poly * K_per

        # Remaining features (if any)
        if X.shape[1] > 1:
            X_rem = X[:, 1:]
            X2_rem = X2[:, 1:]
            K_lin = self.beta_lin ** 2 * (X_rem @ X2_rem.T)
        else:
            K_lin = 0

        return K_nonlin + K_lin

    def Kdiag(self, X):
        # Diagonal of the kernel matrix
        x1 = X[:, 0:1]
        dot_poly = self.gamma_poly * (x1 * x1) + self.c_poly
        K_poly_diag = dot_poly ** 2
        K_per_diag = self.sigma_per ** 2 * np.ones_like(K_poly_diag)
        K_nonlin_diag = K_poly_diag.flatten() * K_per_diag.flatten()

        if X.shape[1] > 1:
            K_lin_diag = self.beta_lin ** 2 * np.sum(X[:, 1:] ** 2, axis=1)
        else:
            K_lin_diag = 0

        return K_nonlin_diag + K_lin_diag


class OnlineFITCGP:
    def __init__(self, input_dim, kernel, inducing_points, noise_var=0.1):
        self.kernel = kernel
        self.Z = inducing_points.copy()  # inducing_points must be array, not int
        self.noise_var = noise_var
        self.input_dim = input_dim

        # Initialize parameters
        self.Kmm = self.kernel.K(self.Z)
        self.Lmm = np.linalg.cholesky(self.Kmm + 1e-6 * np.eye(len(self.Z)))
        self.Kmm_inv = np.linalg.inv(self.Kmm + 1e-6 * np.eye(len(self.Z)))

        self.m = np.zeros(len(self.Z))  # Posterior mean
        self.S = self.Kmm.copy()  # Posterior covariance

        # Sufficient statistics
        self.Q = np.zeros_like(self.Kmm)
        self.b = np.zeros(len(self.Z))
        self.N = 0

    def update(self, X_new, y_new):
        Knm = self.kernel.K(X_new, self.Z)  # shape (N,M)
        kmn = Knm.T  # shape (M,N)

        Lambda_inv = self.kernel.Kdiag(X_new) - np.diag(Knm @ self.Kmm_inv @ kmn) + self.noise_var
        Lambda_inv = Lambda_inv.reshape(-1, 1)  # shape (N,1)

        self.Q += kmn @ (Knm / Lambda_inv)  # (M,N)@(N,M) => (M,M)

        ratio = y_new.reshape(-1, 1) / Lambda_inv  # (N,1) / (N,1) => (N,1)
        ratio = ratio.flatten()  # (N,)
        self.b += kmn @ ratio  # (M,N)@(N,) => (M,)

        self.N += len(y_new)

        self.S = np.linalg.inv(self.Kmm_inv + self.Q / self.noise_var)
        self.m = self.S @ (self.b / self.noise_var)

    def predict(self, X_test):
        Ksm = self.kernel.K(X_test, self.Z)
        Kss = self.kernel.Kdiag(X_test)

        mean = Ksm @ self.m
        var = Kss - np.diag(Ksm @ (self.Kmm - self.S) @ Ksm.T)

        return mean, var

    def optimize(self):
        def neg_log_marginal_likelihood(params):
            kernel_name = type(self.kernel).__name__

            if kernel_name == 'ParabolicPeriodicLinear':
                self.kernel.gamma_poly[:] = params[0]
                self.kernel.c_poly[:] = params[1]
                self.kernel.sigma_per[:] = params[2]
                self.kernel.length_per[:] = params[3]
                self.kernel.period[:] = params[4]
                self.kernel.beta_lin[:] = params[5]

            elif kernel_name == 'CompositeParabolicPeriodicARDRBF':
                # Update parameters with exact names from the kernel
                self.kernel.gamma_poly[:] = params[0]
                self.kernel.c_poly[:] = params[1]
                self.kernel.sigma_per[:] = params[2]
                self.kernel.length_per[:] = params[3]
                self.kernel.period[:] = params[4]

                if hasattr(self.kernel, 'rbf_lengths'):
                    # For input_dim > 1 case
                    self.kernel.rbf_lengths[:] = params[5]
                    self.kernel.rbf_sigma[:] = params[6]
                # else: kernel was initialized with input_dim=1

            else:
                raise ValueError(f"Unsupported kernel: {kernel_name}")

            # Recompute matrices
            self.Kmm = self.kernel.K(self.Z)
            self.Kmm += 1e-6 * np.eye(len(self.Z))
            self.Lmm = np.linalg.cholesky(self.Kmm)
            self.Kmm_inv = np.linalg.inv(self.Kmm)

            return np.sum(self.m ** 2)

        # Initial params depend on kernel
        kernel_name = type(self.kernel).__name__

        if kernel_name == 'ParabolicPeriodicLinear':
            init_params = np.array([
                self.kernel.gamma_poly.values[0],
                self.kernel.c_poly.values[0],
                self.kernel.sigma_per.values[0],
                self.kernel.length_per.values[0],
                self.kernel.period.values[0],
                self.kernel.beta_lin.values[0],
            ])

        elif kernel_name == 'CompositeParabolicPeriodicARDRBF':
            if hasattr(self.kernel, 'rbf_lengths'):
                # For input_dim > 1 case
                init_params = np.array([
                    self.kernel.gamma_poly.values[0],
                    self.kernel.c_poly.values[0],
                    self.kernel.sigma_per.values[0],
                    self.kernel.length_per.values[0],
                    self.kernel.period.values[0],
                    self.kernel.rbf_lengths.values[0],  # First lengthscale
                    self.kernel.rbf_sigma.values[0],
                ])
            else:
                # For input_dim = 1 case
                init_params = np.array([
                    self.kernel.gamma_poly.values[0],
                    self.kernel.c_poly.values[0],
                    self.kernel.sigma_per.values[0],
                    self.kernel.length_per.values[0],
                    self.kernel.period.values[0],
                ])

        else:
            raise ValueError(f"Unsupported kernel: {kernel_name}")

        res = minimize(neg_log_marginal_likelihood, init_params, method='L-BFGS-B',
                       bounds=[(1e-5, None)] * len(init_params))  # Add positive bounds

        # Update kernel params after optimization
        if kernel_name == 'ParabolicPeriodicLinear':
            self.kernel.gamma_poly[:] = res.x[0]
            self.kernel.c_poly[:] = res.x[1]
            self.kernel.sigma_per[:] = res.x[2]
            self.kernel.length_per[:] = res.x[3]
            self.kernel.period[:] = res.x[4]
            self.kernel.beta_lin[:] = res.x[5]

        elif kernel_name == 'CompositeParabolicPeriodicARDRBF':
            self.kernel.gamma_poly[:] = res.x[0]
            self.kernel.c_poly[:] = res.x[1]
            self.kernel.sigma_per[:] = res.x[2]
            self.kernel.length_per[:] = res.x[3]
            self.kernel.period[:] = res.x[4]

            if hasattr(self.kernel, 'rbf_lengths'):
                self.kernel.rbf_lengths[:] = res.x[5]
                self.kernel.rbf_sigma[:] = res.x[6]

        # Recompute kernel matrices
        self.Kmm = self.kernel.K(self.Z)
        self.Kmm += 1e-6 * np.eye(len(self.Z))
        self.Lmm = np.linalg.cholesky(self.Kmm)
        self.Kmm_inv = np.linalg.inv(self.Kmm)


def OnlineFITCGPCaller(X_train, y_train, kernel, MAX_INDUCING, INITIAL_BATCH_SIZE, INCREMENT_SIZE, noise_var=0.1):
    n_samples, n_features = X_train.shape
    X_initial = X_train[:INITIAL_BATCH_SIZE]
    y_initial = y_train[:INITIAL_BATCH_SIZE]

    # Initialize online GP with inducing points array, not integer
    online_gp = OnlineFITCGP(input_dim=n_features,
                             kernel=kernel,
                             inducing_points=X_initial,
                             noise_var=noise_var)

    # Warm start
    online_gp.update(X_train[:INITIAL_BATCH_SIZE], y_train[:INITIAL_BATCH_SIZE])
    mse_list = []
    r2_list = []
    epoch_list = []
    OPTIMIZE_EVERY = 100  # optimize every 100 samples
    for i in range(INITIAL_BATCH_SIZE, len(X_train), INCREMENT_SIZE):
        X_new = X_train[i:i + INCREMENT_SIZE]
        y_new = y_train[i:i + INCREMENT_SIZE]
        online_gp.update(X_new, y_new)

        # Periodic optimization
        if (i - INITIAL_BATCH_SIZE) % OPTIMIZE_EVERY == 0:
            print(f"Optimizing kernel hyperparameters at sample {i}...")
            online_gp.optimize()

        # if i % 50 == 0:
        y_pred, _ = online_gp.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        r2 = r2_score(y_new, y_pred)
        mse_list.append(mse)
        r2_list.append(r2)
        epoch_list.append(i)  # <--- record the exact sample count

        print(f"Processed {i} samples: MSE={mse:.4f}, R²={r2:.4f}")
    return online_gp, r2_list, mse_list, epoch_list


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 1000
    n_features = 2
    noise_level = 0.1
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1  # This determines the period.
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound, stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # GP settings and initial hyperparameters.
    INITIAL_BATCH_SIZE = 100
    INCREMENT_SIZE = 20
    DECAY_GAMMA = .99
    STRETCH_FACTOR = 1
    INITIAL_KERNEL = "rbf"
    MAX_INDUCING = 100  # maximum number of inducing points to retain.

    kernel = ParabolicPeriodicLinear(input_dim=n_features)
    online_gp, r2_list, mse_list, epoch_list  = OnlineFITCGPCaller(X_train, y_train,kernel,MAX_INDUCING,INITIAL_BATCH_SIZE, INCREMENT_SIZE, noise_var=noise_level)

    y_pred, pred_var = online_gp.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    print(f"\nFinal Performance: MSE={final_mse:.4f}, R²={final_r2:.4f}")


