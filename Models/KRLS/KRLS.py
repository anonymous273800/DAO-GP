import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
from sklearn.utils import shuffle

# https://github.com/minhearn/Kernel-Recursive-Least-Squares

class KRLS:
    def __init__(self, x_dim, kernel, criterion=1e-6, sigma=0.2):
        self.x_dim = x_dim
        self.kernel = kernel
        self.criterion = criterion
        self.sigma = sigma
        self.dictionary = np.zeros((0, x_dim))
        self.K_inv = None
        self.P_inv = None
        self.params = None

    def feature(self, x):
        X1 = self.dictionary
        X2 = x.reshape(1, -1)
        return self.kernel(X1, X2, self.sigma).flatten()

    def update(self, x, y):
        x = x.reshape(1, -1)
        if self.dictionary.shape[0] == 0:
            self.dictionary = np.copy(x)
            self.K_inv = np.array([[1.0]])
            self.P_inv = np.array([[1.0]])
            self.params = np.array([y])
            return

        beta = self.feature(x.flatten())
        yhat = np.dot(self.params, beta)
        e = y - yhat

        a = self.K_inv @ beta
        delta = 1 - beta @ a

        if delta > self.criterion:
            self.dictionary = np.vstack([self.dictionary, x])
            delta_inv = 1.0 / delta

            new_K_inv = np.zeros((self.K_inv.shape[0] + 1, self.K_inv.shape[1] + 1))
            new_K_inv[:-1, :-1] = self.K_inv + delta_inv * np.outer(a, a)
            new_K_inv[:-1, -1] = -delta_inv * a
            new_K_inv[-1, :-1] = -delta_inv * a
            new_K_inv[-1, -1] = delta_inv
            self.K_inv = new_K_inv

            d = self.P_inv.shape[0]
            new_P_inv = np.zeros((d + 1, d + 1))
            new_P_inv[:-1, :-1] = self.P_inv
            new_P_inv[-1, -1] = 1.0
            self.P_inv = new_P_inv

            self.params = self.params - a * e * delta_inv
            self.params = np.append(self.params, e * delta_inv)
        else:
            Pa = self.P_inv @ a
            q = Pa / (1 + a @ Pa)
            self.P_inv -= np.outer(q, Pa)
            self.params += e * (self.K_inv @ q)

    def predict(self, x):
        if self.dictionary.shape[0] == 0:
            return 0.0
        beta = self.feature(x)
        return np.dot(self.params, beta)


# --- Simple RBF Kernel ---
def rbf_kernel(X1, X2, sigma):
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return np.exp(-dists / (2 * sigma ** 2))


# --- Trainer ---
def kernelized_RLS(X_train, y_train, lambdaa, sigma, kernel):
    n_samples, n_features = X_train.shape
    model = KRLS(x_dim=n_features, kernel=kernel, criterion=lambdaa, sigma=sigma)
    r2_list = []
    mse_list = []
    epoch_list = []
    for i, (xi, yi) in enumerate(zip(X_train, y_train)):
        model.update(xi, yi)
        if (i + 1) % 50 == 0 or i == len(X_train) - 1:
            y_pred = np.array([model.predict(xi) for xi in X_train])
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            r2_list.append(r2)
            mse_list.append(mse)
            epoch_list.append(i+1)
            print(f"[{i + 1}] Dict size: {len(model.dictionary):4d} | R²: {r2:.4f} | MSE: {mse:.4f}")
    return model, r2_list, mse_list, epoch_list



# --- Run Experiment ---
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    n_samples = 2000
    n_features = 1
    noise_level = 0.1  # reduced noise
    lower_bound = -5
    upper_bound = 5
    stretch_factor = 1   # more wave cycles = easier to learn

    # Load and split data
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level,
                                     lower_bound, upper_bound, stretch_factor=stretch_factor)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # KRLS training
    lambdaa = 1e-6  # low ALD threshold to allow dictionary growth
    sigma = 0.3

    # Option #1 Kernel = RBF (used with 2D)
    # krls_kernel = rbf_kernel

    # Option #2 composite_parabolic_periodic_ard_rbf (used with high D)
    from Models.KRLS import KRLSKernels
    def custom_kernel(X1, X2, sigma):
        return KRLSKernels.composite_parabolic_periodic_ard_rbf(
            X1, X2,
            gamma_poly=1.0, c_poly=1.0, sigma_per=1.0,
            length_per=1.0, period=1.0,
            rbf_lengths=np.ones(X1.shape[1] - 1) if X1.shape[1] > 1 else np.array([]),
            rbf_sigma=1.0
        )
    krls_kernel = custom_kernel
    model, r2_list, mse_list, epoch_list = kernelized_RLS(X_train, y_train, lambdaa, sigma, krls_kernel)

    # Predict
    y_pred = np.array([model.predict(xi) for xi in X_test])

    # Evaluate
    print("R²:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("Final dictionary size:", len(model.dictionary))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, y_test, label='True', color='blue', alpha=0.6)
    plt.scatter(X_test, y_pred, label='KRLS Predicted', color='red', alpha=0.6)
    plt.title("KRLS with RBF Kernel")
    plt.legend()
    plt.grid(True)
    plt.show()
