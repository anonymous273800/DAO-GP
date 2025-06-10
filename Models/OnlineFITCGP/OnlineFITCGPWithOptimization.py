import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
import warnings
warnings.filterwarnings("ignore")
import GPy
from scipy.optimize import minimize

# --- Custom Kernel ---
class ParabolicPeriodicLinear(GPy.kern.Kern):
    def __init__(self, input_dim, active_dims=None, name='ParabolicPeriodicLinear',
                 gamma_poly=1.0, c_poly=1.0, sigma_per=1.0, length_per=1.0, period=1.0, beta_lin=1.0):
        super().__init__(input_dim, active_dims, name)
        self.gamma_poly = GPy.core.Param('gamma_poly', gamma_poly)
        self.c_poly = GPy.core.Param('c_poly', c_poly)
        self.sigma_per = GPy.core.Param('sigma_per', sigma_per)
        self.length_per = GPy.core.Param('length_per', length_per)
        self.period = GPy.core.Param('period', period)
        self.beta_lin = GPy.core.Param('beta_lin', beta_lin)
        self.link_parameters(self.gamma_poly, self.c_poly, self.sigma_per,
                             self.length_per, self.period, self.beta_lin)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        x1 = X[:, 0:1]
        x2 = X2[:, 0:1]
        dot_poly = self.gamma_poly * (x1 @ x2.T) + self.c_poly
        K_poly = dot_poly ** 2
        diff = np.abs(x1 - x2.T)
        K_per = self.sigma_per ** 2 * np.exp(-2 * (np.sin(np.pi * diff / self.period) ** 2) / (self.length_per ** 2))
        K_nonlin = K_poly * K_per
        K_lin = 0
        if X.shape[1] > 1:
            X_rem = X[:, 1:]
            X2_rem = X2[:, 1:]
            K_lin = self.beta_lin ** 2 * (X_rem @ X2_rem.T)
        return K_nonlin + K_lin

    def Kdiag(self, X):
        x1 = X[:, 0:1]
        dot_poly = self.gamma_poly * (x1 * x1) + self.c_poly
        K_poly_diag = dot_poly ** 2
        K_per_diag = self.sigma_per ** 2 * np.ones_like(K_poly_diag)
        K_nonlin_diag = K_poly_diag.flatten() * K_per_diag.flatten()
        K_lin_diag = 0
        if X.shape[1] > 1:
            K_lin_diag = self.beta_lin ** 2 * np.sum(X[:, 1:] ** 2, axis=1)
        return K_nonlin_diag + K_lin_diag


# --- Online FITC-GP Model ---
class OnlineFITCGP:
    def __init__(self, input_dim, kernel, inducing_points, noise_var=0.1):
        self.kernel = kernel
        self.Z = inducing_points.copy()
        self.noise_var = noise_var
        self.input_dim = input_dim
        self.Kmm = self.kernel.K(self.Z)
        self.Lmm = np.linalg.cholesky(self.Kmm + 1e-6 * np.eye(len(self.Z)))
        self.Kmm_inv = np.linalg.inv(self.Kmm + 1e-6 * np.eye(len(self.Z)))
        self.m = np.zeros(len(self.Z))
        self.S = self.Kmm.copy()
        self.Q = np.zeros_like(self.Kmm)
        self.b = np.zeros(len(self.Z))
        self.N = 0

    def update(self, X_new, y_new):
        Knm = self.kernel.K(X_new, self.Z)
        kmn = Knm.T
        Lambda_inv = self.kernel.Kdiag(X_new) - np.diag(Knm @ self.Kmm_inv @ kmn) + self.noise_var
        Lambda_inv = Lambda_inv.reshape(-1, 1)
        self.Q += kmn @ (Knm / Lambda_inv)
        ratio = y_new.reshape(-1, 1) / Lambda_inv
        self.b += kmn @ ratio.flatten()
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
            self.kernel.gamma_poly[:] = params[0]
            self.kernel.c_poly[:] = params[1]
            self.kernel.sigma_per[:] = params[2]
            self.kernel.length_per[:] = params[3]
            self.kernel.period[:] = params[4]
            self.Kmm = self.kernel.K(self.Z) + 1e-6 * np.eye(len(self.Z))
            self.Lmm = np.linalg.cholesky(self.Kmm)
            self.Kmm_inv = np.linalg.inv(self.Kmm)
            return np.sum(self.m ** 2)

        init_params = np.array([
            self.kernel.gamma_poly.values[0],
            self.kernel.c_poly.values[0],
            self.kernel.sigma_per.values[0],
            self.kernel.length_per.values[0],
            self.kernel.period.values[0],
        ])
        res = minimize(neg_log_marginal_likelihood, init_params, method='L-BFGS-B',
                       bounds=[(1e-5, None)] * len(init_params))
        self.kernel.gamma_poly[:] = res.x[0]
        self.kernel.c_poly[:] = res.x[1]
        self.kernel.sigma_per[:] = res.x[2]
        self.kernel.length_per[:] = res.x[3]
        self.kernel.period[:] = res.x[4]
        self.Kmm = self.kernel.K(self.Z) + 1e-6 * np.eye(len(self.Z))
        self.Lmm = np.linalg.cholesky(self.Kmm)
        self.Kmm_inv = np.linalg.inv(self.Kmm)


# --- Main Online Training Logic ---
def OnlineFITCGPCaller(X_train, y_train, kernel, MAX_INDUCING, INITIAL_BATCH_SIZE, INCREMENT_SIZE,
                       noise_var=0.1, optimize_every=50):
    n_samples, n_features = X_train.shape
    X_initial = X_train[:INITIAL_BATCH_SIZE]
    y_initial = y_train[:INITIAL_BATCH_SIZE]

    online_gp = OnlineFITCGP(input_dim=n_features,
                             kernel=kernel,
                             inducing_points=X_initial,
                             noise_var=noise_var)
    online_gp.update(X_initial, y_initial)
    mse_list = []
    r2_list = []
    epoch_list = []

    for i in range(INITIAL_BATCH_SIZE, len(X_train), INCREMENT_SIZE):
        X_new = X_train[i:i + INCREMENT_SIZE]
        y_new = y_train[i:i + INCREMENT_SIZE]
        online_gp.update(X_new, y_new)

        if (i - INITIAL_BATCH_SIZE) % optimize_every == 0:
            print(f"Optimizing kernel hyperparameters at sample {i}...")
            online_gp.optimize()

        y_pred, _ = online_gp.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        r2 = r2_score(y_new, y_pred)
        mse_list.append(mse)
        r2_list.append(r2)
        epoch_list.append(i)
        print(f"Processed {i} samples: MSE={mse:.4f}, R²={r2:.4f}")

    return online_gp, r2_list, mse_list, epoch_list


# --- Main Execution ---
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 1000
    n_features = 2
    noise_level = 0.1
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound, stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    INITIAL_BATCH_SIZE = 100
    INCREMENT_SIZE = 20
    MAX_INDUCING = 100
    OPTIMIZE_EVERY = 50

    kernel = ParabolicPeriodicLinear(
        input_dim=n_features,
        gamma_poly=1.0,
        c_poly=1.0,
        sigma_per=1.0,
        length_per=1.0,
        period=stretch_factor,
        beta_lin=1.0
    )

    online_gp, r2_list, mse_list, epoch_list = OnlineFITCGPCaller(
        X_train, y_train, kernel,
        MAX_INDUCING, INITIAL_BATCH_SIZE, INCREMENT_SIZE,
        noise_var=noise_level, optimize_every=OPTIMIZE_EVERY
    )

    y_pred, pred_var = online_gp.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    print(f"\nFinal Performance: MSE={final_mse:.4f}, R²={final_r2:.4f}")
