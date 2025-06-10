# LIMITATION OF OKRR, it stores all past observations
'''
Yes, while Online Kernel Ridge Regression (OKRR) is designed for efficiently incorporating new data into the model, it does have a limitation related to storing all past data. OKRR, unlike standard Kernel Ridge Regression (KRR), updates its model with new data points incrementally without requiring the entire historical dataset. This makes it computationally advantageous for online learning tasks. However, OKRR constructs a kernel using only the N most recent time points, which can be a limitation when dealing with data exhibiting seasonal or cyclical patterns. In these cases, information from more distant past time points might be relevant, and OKRR's focus on recent data could lead to suboptimal forecasts.
Here's a more detailed breakdown:
OKRR's Efficiency:
OKRR's strength lies in its ability to adapt the model to new data without retraining on the entire dataset. This makes it suitable for situations where data streams in continuously, and maintaining a large historical record is impractical.
The Limitation:
OKRR's kernel is built using only a subset of the most recent data, which is effective for scenarios where data is relatively stationary or exhibits sudden regime changes. However, when dealing with time series data that have seasonal or cyclical patterns, this limitation can be problematic.
Seasonal/Cyclical Data:
In these cases, data from previous seasons, weeks, or months can provide valuable insights. Focusing solely on the most recent data might lead to overlooking these long-term patterns.
Local OKRR (LOKRR):
To address this limitation, a variant called Local Online Kernel Ridge Regression (LOKRR) has been proposed. LOKRR constructs separate kernels for different time periods (e.g., different times of day or different days of the week), allowing the model to capture more diverse patterns, including seasonal and cyclical ones.
Local online kernel ridge regression for forecasting of urban travel ...
2.5. Local online kernel ridge regression. OKRR has the advantage over standard KRR that it can incorporate new information in the...

ScienceDirect.com

Local online kernel ridge regression for forecasting of urban travel ...
However, it does not address the issue that only very recent traffic patterns are included in the kernel, in this case 15 days. It...

UCL Discovery

Ridge Regression: Challenges and Limitations - LinkedIn
Mar 15, 2023 — This method can lead to less interpretability than simpler models, as coefficients are shrunk. While it helps in deali...

LinkedIn

Show all

'''


import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin

# ——— Kernel definitions ———

def parabolic_periodic_kernel(X1, X2,
                              gamma_poly, c_poly,
                              sigma_per, length_per, period):
    x1 = X1[:, :1]
    x2 = X2[:, :1]
    parabolic = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    diff      = np.abs(x1 - x2.T)
    periodic  = np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))
    return sigma_per**2 * parabolic * periodic

def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
    if X1.shape[1] > 1:
        X1r = X1[:, 1:]
        X2r = X2[:, 1:]
        diff   = X1r[:, None, :] - X2r[None, :, :]
        sqdist = np.sum((diff**2) / (length_scales**2), axis=2)
        return sigma_f**2 * np.exp(-0.5 * sqdist)
    return np.zeros((X1.shape[0], X2.shape[0]))

def composite_parabolic_periodic_ard_rbf(X1, X2,
                                         gamma_poly, c_poly,
                                         sigma_per, length_per, period,
                                         rbf_lengths, rbf_sigma):
    K1 = parabolic_periodic_kernel(
        X1, X2, gamma_poly, c_poly,
        sigma_per, length_per, period
    )
    K2 = ard_rbf_kernel(X1, X2, rbf_lengths, rbf_sigma)
    return K1 + K2

# ——— Online Kernel Ridge Regressor ———

class OnlineKernelRidge:
    def __init__(self, kernel_fn, alpha):
        self.kernel_fn = kernel_fn
        self.alpha     = alpha
        self.X         = None
        self.A_inv     = None
        self.dual      = None

    def partial_fit(self, X_new, y_new):
        # Accept either a batch or a single sample
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)
        if X_new.shape[0] > 1:
            # process each sample in turn
            for x_i, y_i in zip(X_new, y_new):
                self.partial_fit(x_i, y_i)
            return

        # single-sample update
        x = X_new  # shape (1, d)
        y = float(y_new[0])

        if self.X is None:
            # initialize on first sample
            k11 = float(np.atleast_2d(self.kernel_fn(x, x))[0, 0])
            c   = k11 + self.alpha
            self.A_inv = np.array([[1.0 / c]])
            self.dual  = np.array([y / c])
            self.X     = x
            return

        # incremental Sherman–Morrison update
        X_old = self.X
        A     = self.A_inv
        α_old = self.dual

        k_vec = self.kernel_fn(X_old, x).ravel()                    # shape (n,)
        k_nn  = float(np.atleast_2d(self.kernel_fn(x, x))[0, 0])    # scalar

        c    = k_nn + self.alpha
        Akn  = A.dot(k_vec)                                         # shape (n,)
        eps = 1e-10
        beta = max(c - k_vec.dot(Akn), eps)                                   # scalar

        # update inverse
        TL = A + np.outer(Akn, Akn) / beta
        TR = -Akn / beta
        BL = -Akn.reshape(1, -1) / beta
        BR = np.array([[1.0 / beta]])
        self.A_inv = np.block([[TL, TR.reshape(-1,1)], [BL, BR]])

        # update dual weights
        delta     = k_vec.dot(α_old) - y
        α1        = α_old + (Akn / beta) * delta
        α2        = -delta / beta
        self.dual = np.concatenate([α1, [α2]])

        # append new sample to memory
        self.X = np.vstack([X_old, x])

    def predict(self, X_test):
        Kt = self.kernel_fn(X_test, self.X)
        return Kt.dot(self.dual)

# ——— Streaming training (no accumulation) ———

def okrr(X_train, y_train, kernel, alpha, batch_size):
    model   = OnlineKernelRidge(kernel_fn=kernel, alpha=alpha)
    r2_list = []
    mse_list= []
    epoch_list = []

    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        Xb  = X_train[start:end]
        yb  = y_train[start:end]

        # train on this minibatch
        model.partial_fit(Xb, yb)

        # evaluate on the same minibatch, then forget it
        y_pred = model.predict(Xb)
        r2  = r2_score(yb, y_pred)
        mse = mean_squared_error(yb, y_pred)
        print(f"OKRR Batch {start:4d}-{end:4d} → R² = {r2:.4f}, MSE = {mse:.4f}")

        r2_list.append(r2)
        mse_list.append(mse)
        epoch_list.append(end)

    return model, r2_list, mse_list, epoch_list


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

    INCREMENT_SIZE = 20
    # dynamic kernel hyper-parameters
    n_samples, n_features = X_train.shape
    kernel_params = {
        "gamma_poly": 0.5,
        "c_poly":     1.0,
        "sigma_per":  1.0,
        "length_per": 1.0,
        "period":     2.0,
        "rbf_lengths": np.ones(n_features - 1),
        "rbf_sigma":  1.0,
    }

    # single callable kernel
    def kernel(X1, X2):
        return composite_parabolic_periodic_ard_rbf(X1, X2, **kernel_params)

    # train on minibatches and evaluate per-batch
    model, r2s, mses, epoch_list = okrr(X_train, y_train,kernel=kernel,alpha=0.001, batch_size=INCREMENT_SIZE)

    # final hold-out evaluation
    y_test_pred = model.predict(X_test)
    print(f"\nFinal Test R²:  {r2_score(y_test, y_test_pred):.4f}")
    print(f"Final Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
