# Limitations of KPA:
'''
There is no support vector pruning.
The C parameter is fixed, not adaptive.
The kernel is predefined and fixed (not learned or adapted).
All support vectors that cause an update are retained forever.
It's designed as a theoretically clean, mistake-driven online learning algorithm, not a memory-bounded or drift-aware learner.
This is consistent with:
Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006). Online Passive-Aggressive Algorithms. JMLR.
'''

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
import warnings
warnings.filterwarnings("ignore")

# ------------------ Composite Kernel ------------------

def parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period):
    x1 = X1[:, 0:1]
    x2 = X2[:, 0:1]
    parabolic = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    diff = np.abs(x1 - x2.T)
    periodic = np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))
    return sigma_per ** 2 * parabolic * periodic

def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
    if X1.shape[1] > 1:
        X1_rest = X1[:, 1:]
        X2_rest = X2[:, 1:]
        diff = X1_rest[:, None, :] - X2_rest[None, :, :]
        sqdist = np.sum((diff ** 2) / (length_scales ** 2), axis=2)
        return sigma_f ** 2 * np.exp(-0.5 * sqdist)
    return np.zeros((X1.shape[0], X2.shape[0]))

def composite_parabolic_periodic_ard_rbf(X1, X2,
                                         gamma_poly=1.0, c_poly=1.0, sigma_per=1.0,
                                         length_per=1.0, period=1.0,
                                         rbf_lengths=None, rbf_sigma=1.0):
    if rbf_lengths is None:
        rbf_lengths = np.ones(X1.shape[1] - 1) if X1.shape[1] > 1 else np.array([])
    K_par_per = parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period)
    K_ard = ard_rbf_kernel(X1, X2, rbf_lengths, rbf_sigma)
    return K_par_per + K_ard

# ------------------ KPA Regressor ------------------

class KPARegressor:
    def __init__(self, kernel_fn, C=1.0, epsilon=0.1):
        self.kernel_fn = kernel_fn
        self.C = C
        self.epsilon = epsilon
        self.support_vectors = []
        self.alpha = []
        self.y_true = []  # actual targets of support vectors

    def predict(self, x):
        if not self.support_vectors:
            return np.zeros(x.shape[0])
        K = self.kernel_fn(x, np.array(self.support_vectors))
        return K @ np.array(self.alpha)

    def partial_fit(self, X_t, y_t):
        for i in range(X_t.shape[0]):
            x_i = X_t[i:i+1]
            y_i = y_t[i]
            y_pred = self.predict(x_i)[0]
            loss = max(0, abs(y_pred - y_i) - self.epsilon)
            if loss > 0:
                K_tt = self.kernel_fn(x_i, x_i)[0, 0]
                tau = min(self.C, loss / (K_tt + 1e-10))
                self.alpha.append(tau * np.sign(y_i - y_pred))
                self.support_vectors.append(x_i[0])
                self.y_true.append(y_i)

# ------------------ Training Function ------------------

def kpa(X_train, y_train, kernel, batch_size, C=1.0, epsilon=0.1):
    n_samples, n_features = X_train.shape
    model = KPARegressor(kernel_fn=kernel, C=C, epsilon=epsilon)
    r2_list = []
    mse_list = []
    epoch_list = []
    for i in range(0, n_samples, batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        model.partial_fit(X_batch, y_batch)

        # Predict on seen support vectors so far
        if model.support_vectors:
            X_seen = np.array(model.support_vectors)
            y_seen = np.array(model.y_true)
            y_pred = model.predict(X_seen)
            r2 = r2_score(y_seen, y_pred)
            mse = mean_squared_error(y_seen, y_pred)
            print(f"Trained on {i + len(X_batch):4d} samples -> R² = {r2:.4f}, MSE = {mse:.4f}")
            r2_list.append(r2)
            mse_list.append(mse)
            epoch_list.append(i + len(X_batch))
    return model, r2_list, mse_list, epoch_list

# ------------------ Run Example ------------------

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 2000
    n_features = 1
    noise_level = 0.01
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1  # This determines the period.
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound,
                                     stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Kernel setup
    def custom_kernel(X1, X2):
        return composite_parabolic_periodic_ard_rbf(
            X1, X2,
            gamma_poly=1.0, c_poly=1.0, sigma_per=1.0,
            length_per=1.0, period=1.0,
            rbf_lengths=np.ones(X1.shape[1] - 1) if X1.shape[1] > 1 else np.array([]),
            rbf_sigma=1.0
        )

    INCREMENT_SIZE = 20
    model, r2_list, mse_list, epoch_list = kpa(X_train, y_train, kernel=custom_kernel, batch_size=INCREMENT_SIZE, C=1, epsilon=0.1)

    # Final evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nFinal Test R² = {r2:.4f}")
    print(f"Final Test MSE = {mse:.4f}")
