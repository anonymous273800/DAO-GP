import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, Kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import your sinusoidal dataset function.
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin


# ---------------------------
# Custom kernel: FixedDimensionKernel
# ---------------------------
class FixedDimensionKernel(Kernel):
    """
    A wrapper that applies a given base kernel only to specified dimensions of the input.
    """
    def __init__(self, base_kernel, dims):
        """
        Parameters:
          base_kernel : A scikit-learn kernel instance (e.g. RBF, ExpSineSquared)
          dims : int or list of int
            The indices of the dimensions on which to apply the kernel.
        """
        self.base_kernel = base_kernel
        if isinstance(dims, int):
            self.dims = [dims]
        else:
            self.dims = dims

    def __call__(self, X, Y=None, eval_gradient=False):
        X_sub = X[:, self.dims]
        if Y is not None:
            Y_sub = Y[:, self.dims]
        else:
            Y_sub = None
        # Forward the call to the base kernel.
        if eval_gradient:
            K, grad = self.base_kernel(X_sub, Y_sub, eval_gradient=True)
            return K, grad
        else:
            return self.base_kernel(X_sub, Y_sub, eval_gradient=False)

    def diag(self, X):
        X_sub = X[:, self.dims]
        return self.base_kernel.diag(X_sub)

    def is_stationary(self):
        return self.base_kernel.is_stationary

    def __repr__(self):
        return f"FixedDimensionKernel({self.base_kernel}, dims={self.dims})"

    # For full compatibility with hyperparameter optimization, delegate theta and bounds:
    @property
    def theta(self):
        return self.base_kernel.theta

    @theta.setter
    def theta(self, val):
        self.base_kernel.theta = val

    @property
    def bounds(self):
        return self.base_kernel.bounds

    def clone_with_theta(self, theta):
        new_base = self.base_kernel.clone_with_theta(theta)
        return FixedDimensionKernel(new_base, self.dims)

# ---------------------------
# Main section
# ---------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Dataset parameters
    n_samples = 1000
    n_features = 10    # Data is 10-dimensional, with sinusoidal behavior in the first dimension.
    noise_level = 0.1
    lower_bound = -2
    upper_bound = 2
    stretch_factor = 1  # This controls the period in DS001_Sinusoidal

    # Load the sinusoidal dataset
    X, y = DS001Sin.DS001_Sinusoidal(n_samples, n_features, noise_level, lower_bound, upper_bound, stretch_factor=stretch_factor)

    # Standardize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the kernel:
    # Use an ExpSineSquared (periodic) kernel on the first dimension.
    period = stretch_factor  # For the periodic component
    per_kernel = FixedDimensionKernel(
        ExpSineSquared(length_scale=1.0,
                       periodicity=period,
                       length_scale_bounds=(1e-2, 1e2),
                       periodicity_bounds=(1e-2, 1e2)),
        dims=0
    )

    # Use an RBF kernel on the remaining dimensions.
    rbf_kernel = FixedDimensionKernel(
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
        dims=list(range(1, n_features))
    )

    # White noise kernel
    noise_kernel = WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-1, 1e1))

    # Combine the kernels additively
    kernel = per_kernel + rbf_kernel + noise_kernel

    print("Composite kernel:", kernel)

    # Create the Gaussian Process Regressor.
    # Option 1: Allow hyperparameter optimization:
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    # Option 2 (if you get gradient errors): disable optimization by setting optimizer=None
    # gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, random_state=42)

    # Fit the batch GP on the training data
    gp.fit(X_train, y_train)

    # Evaluate on the training set
    y_train_pred, y_train_std = gp.predict(X_train, return_std=True)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Batch GP - Train MSE: {train_mse:.4f}, Train R^2: {train_r2:.4f}")

    # Evaluate on the test set
    y_test_pred, y_test_std = gp.predict(X_test, return_std=True)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Batch GP - Test MSE: {test_mse:.4f}, Test R^2: {test_r2:.4f}")

    # (Optional) Plot predictions versus true values along the first dimension
    sorted_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_idx]
    y_test_sorted = y_test[sorted_idx]
    y_pred_sorted = y_test_pred[sorted_idx]
    std_sorted = y_test_std[sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.scatter(X_test_sorted[:, 0], y_test_sorted, color='blue', alpha=0.4, label='Test Data')
    plt.plot(X_test_sorted[:, 0], y_pred_sorted, 'r-', label='GP Mean')
    plt.fill_between(X_test_sorted[:, 0],
                     y_pred_sorted - 2 * std_sorted,
                     y_pred_sorted + 2 * std_sorted,
                     color='r', alpha=0.2, label='±2 std')
    plt.title("Batch Gaussian Process Regression")
    plt.xlabel("Feature 0 (first dimension)")
    plt.ylabel("Standardized Target")
    plt.legend()
    plt.tight_layout()
    plt.show()
