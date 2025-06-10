import numpy as np
from Kernels.KernelsHyperparams import KernelHyperparameterOptimizer
from sklearn.preprocessing import StandardScaler
import os

# def get_decay_matrix(decay_gamma, n_window):
#     # Compute decay weights
#     decay_exps = [decay_gamma ** (n_window - 1 - i) for i in range(n_window)]
#     W = np.diag(decay_exps)
#     W_sqrt = np.diag(np.sqrt(np.diag(W)))
#     W_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(W)))
#
#     return W_sqrt, W_sqrt_inv

# def get_decay_matrix(decay_gamma, X_base_time):
#     current_time = np.max(X_base_time)
#     X_base_time = np.array(X_base_time)
#
#     decay_weights = decay_gamma ** (current_time - X_base_time)
#     decay_weights = np.clip(decay_weights, 1e-12, 1.0)  # avoid zeros
#
#     W_sqrt = np.diag(np.sqrt(decay_weights))
#     W_sqrt_inv = np.diag(1.0 / np.sqrt(decay_weights))
#
#     return W_sqrt, W_sqrt_inv


# def get_decay_matrix_full(decay_gamma, n_window):
#     """
#     Compute a full decay matrix A (diagonal) whose (i,i) element is:
#       A[i,i] = decay_gamma^(n_window-1-i)
#     Older points (lower i) get lower weights.
#     """
#     decay_exps = [decay_gamma ** (n_window - 1 - i) for i in range(n_window)]
#     return np.diag(decay_exps)


# def predict_with_decay(X_window, y_window, K_inv_undecayed, X_star, noise, decay_gamma, decay_option, kernel_func, X_base_time, **kernel_args):
#     X_star = np.atleast_2d(X_star)
#     n_window = len(y_window)
#     y_window = y_window.reshape(-1, 1)
#
#     if decay_option:
#         W_sqrt, W_sqrt_inv = get_decay_matrix(decay_gamma, X_base_time)
#
#         # Apply decay to inverse kernel matrix and y
#         K_inv = W_sqrt_inv @ K_inv_undecayed @ W_sqrt_inv
#         y_weighted = W_sqrt @ y_window
#
#         # Apply decay to k_s
#         k_s = kernel_func(X_window, X_star, **kernel_args)
#         k_s = W_sqrt @ k_s
#     else:
#         # No decay applied
#         K_inv = K_inv_undecayed
#         y_weighted = y_window
#         k_s = kernel_func(X_window, X_star, **kernel_args)
#
#     # Posterior mean
#     mu_star = k_s.T @ K_inv @ y_weighted
#
#     # Posterior covariance
#     K_star_star = kernel_func(X_star, X_star, **kernel_args) + noise * np.eye(len(X_star))
#     cov_star = K_star_star - (k_s.T @ K_inv @ k_s)
#
#     # Ensure symmetric covariance and extract standard deviation
#     cov_star = 0.5 * (cov_star + cov_star.T)
#     diag_cov = np.clip(np.diag(cov_star), 1e-12, None)
#     std_star = np.sqrt(diag_cov)
#
#     return mu_star.ravel(), std_star


# def predict(X_window, y_window, K_inv, X_star, noise, kernel_func, **kernel_args):
#     X_star = np.atleast_2d(X_star)
#     y_window = y_window.reshape(-1, 1)
#     k_s = kernel_func(X_window, X_star, **kernel_args)
#
#     # Posterior mean
#     mu_star = k_s.T @ K_inv @ y_window
#
#     # Posterior covariance
#     K_star_star = kernel_func(X_star, X_star, **kernel_args) + noise * np.eye(len(X_star))
#     cov_star = K_star_star - (k_s.T @ K_inv @ k_s)
#
#     # Ensure symmetric covariance and extract standard deviation
#     cov_star = 0.5 * (cov_star + cov_star.T)
#     diag_cov = np.clip(np.diag(cov_star), 1e-12, None)
#     std_star = np.sqrt(diag_cov)
#
#     return mu_star.ravel(), std_star


def compute_predictive_variance(X_window, y_window, X_base_time, gamma, K_inv_undecayed, x_new, noise, kernel_func, **kernel_args):
    _, std = computeGP(
        X_window=X_window,
        y_window=y_window,
        K_inv=K_inv_undecayed,
        X_star=x_new,
        noise=noise,
        kernel_func=kernel_func,
        **kernel_args
    )

    # _, std = compute_weighted_GP(X_train=X_window, y_train=y_window, X_train_time=X_base_time, gamma=gamma, X_star=x_new, noise=noise, kernel_func=kernel_func, **kernel_args)
    return std[0] ** 2

def reserve_validation_samples(X_inc, y_inc, X_val, y_val, fraction=0.01):
    n_val_samples = max(1, int(np.ceil(fraction * len(X_inc))))
    val_indices = np.random.choice(len(X_inc), size=n_val_samples, replace=False)
    X_val_new = X_inc[val_indices]
    y_val_new = y_inc[val_indices]
    # Remove these samples from the current incremental batch
    X_inc = np.delete(X_inc, val_indices, axis=0)
    y_inc = np.delete(y_inc, val_indices)
    # Append to the existing validation set
    X_val = np.concatenate([X_val, X_val_new], axis=0)
    y_val = np.concatenate([y_val, y_val_new], axis=0)
    return X_inc, y_inc, X_val, y_val


def reserve_validation_samples_drift(X_inc, y_inc, fraction=0.01):
    n_val_samples = max(10, int(np.ceil(fraction * len(X_inc))))  # the minimum is 2 because r^2 at least needs 2 values.
    val_indices = np.random.choice(len(X_inc), size=n_val_samples, replace=False)
    X_val_new = X_inc[val_indices]
    y_val_new = y_inc[val_indices]
    # Remove these samples from the current incremental batch
    X_inc = np.delete(X_inc, val_indices, axis=0)
    y_inc = np.delete(y_inc, val_indices)
    # Append to the existing validation set
    # X_val = np.concatenate([X_val, X_val_new], axis=0)
    # y_val = np.concatenate([y_val, y_val_new], axis=0)
    return X_inc, y_inc, X_val_new, y_val_new


def woodbury_update(K_inv, X_win, x_new, kernel_func, kernel_args, noise, tol=1e-6, rel_tol=1e5):
    x_new = np.atleast_2d(x_new)
    k = kernel_func(X_win, x_new, **kernel_args)
    c = kernel_func(x_new, x_new, **kernel_args)[0, 0] + noise
    d = c - (k.T @ K_inv @ k)[0, 0]

    if d < tol or (c / max(d, 1e-12)) > rel_tol:
        return None

    beta = 1.0 / d
    K_inv_updated_top = K_inv + beta * (K_inv @ k @ k.T @ K_inv)
    K_inv_updated_side = -beta * (K_inv @ k)
    K_inv_new = np.block([
        [K_inv_updated_top, K_inv_updated_side],
        [K_inv_updated_side.T, np.array([[beta]])]
    ])
    return 0.5 * (K_inv_new + K_inv_new.T)


def optimizeKernelHyperparamsGeneric(X_win, y_win, kernel_type, kernel_params, kernel_func):
    optimized_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_win, y_win, kernel_type,kernel_params)
    noise = optimized_dict["noise"]
    kernel_args = {k: v for k, v in optimized_dict.items() if k != "noise"}
    K_win = kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(y_win))
    K_inv_undecayed = np.linalg.inv(K_win)
    return K_win, K_inv_undecayed


def normalize_dataset(X, y):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_norm = X_scaler.fit_transform(X)
    y_norm = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    return X_norm, y_norm


def append_timestamps(X_base_time, num_points, time_value):
    """
    Safely appends `num_points` timestamps with value `time_value` to X_base_time.
    Ensures NumPy array consistency and avoids shape issues.
    """
    X_base_time = np.array(X_base_time)
    new_times = np.full((num_points,), time_value)
    return np.concatenate([X_base_time, new_times])



# def get_decay_matrix_full_timestamp(decay_gamma, X_base_time):
#     """
#     Returns decay matrix A such that A[i, i] = gamma^(current_time - time_i)
#     """
#     X_base_time = np.array(X_base_time)
#     current_time = np.max(X_base_time)
#     decay_weights = decay_gamma ** (current_time - X_base_time)
#     return np.diag(decay_weights)
#
# def predict_with_full_decay(X_window, y_window, K_inv_undecayed, X_star, noise, decay_gamma, decay_option, kernel_func, X_base_time, **kernel_args):
#     X_star = np.atleast_2d(X_star)
#     y_window = y_window.reshape(-1, 1)
#
#     if decay_option:
#         A = get_decay_matrix_full_timestamp(decay_gamma, X_base_time)
#         K = kernel_func(X_window, X_window, **kernel_args)
#         K_weighted = A @ K @ A + noise * np.eye(len(X_window))
#         K_inv = np.linalg.inv(K_weighted)
#
#         y_weighted = A @ y_window
#         k_s = kernel_func(X_window, X_star, **kernel_args)
#         k_weighted = A @ k_s
#     else:
#         K = kernel_func(X_window, X_window, **kernel_args) + noise * np.eye(len(X_window))
#         K_inv = np.linalg.inv(K)
#         y_weighted = y_window
#         k_weighted = kernel_func(X_window, X_star, **kernel_args)
#
#     mu_star = k_weighted.T @ K_inv @ y_weighted
#     K_star_star = kernel_func(X_star, X_star, **kernel_args) + noise * np.eye(len(X_star))
#     cov_star = K_star_star - k_weighted.T @ K_inv @ k_weighted
#
#     cov_star = 0.5 * (cov_star + cov_star.T)
#     diag_cov = np.clip(np.diag(cov_star), 1e-12, None)
#     std_star = np.sqrt(diag_cov)
#
#     return mu_star.ravel(), std_star




def computeGP(X_window, y_window, K_inv, X_star, noise, kernel_func, **kernel_args):
    X_star = np.atleast_2d(X_star)
    y_win = y_window.reshape(-1, 1)

    ks = kernel_func(X_window, X_star, **kernel_args)
    mu_star = ks.T @ K_inv @ y_win
    kss = kernel_func(X_star, X_star, **kernel_args) + noise * np.eye(len(X_star))
    cov_star = kss - ks.T @ K_inv @ ks

    cov_star = 0.5 * (cov_star + cov_star.T)
    diag_cov = np.clip(np.diag(cov_star), 1e-4, None)
    std_star = np.sqrt(diag_cov)

    return mu_star.ravel(), std_star


# def apply_decay_with_time(K_inv_undecayed, DECAY_GAMMA, X_base_time):
#     timestamps = np.array(X_base_time)
#     max_time = np.max(timestamps)
#     # Compute decay weights: w_i = gamma^(max_time - t_i)
#     w = np.array([DECAY_GAMMA ** (max_time - t) for t in timestamps])
#     # Instead of D * K_inv * D, we need to use D_inv * K_inv * D_inv,
#     # where D_inv = diag(1/sqrt(w))
#     D_inv = np.diag(1.0 / np.sqrt(w))
#     return D_inv @ K_inv_undecayed @ D_inv


# def apply_decay_with_time(K_inv_undecayed, DECAY_GAMMA, X_base_time):
#     timestamps = np.array(X_base_time)
#     max_time = np.max(timestamps)
#     # Compute decay weights: w_i = gamma^(max_time - t_i)
#     w = np.array([DECAY_GAMMA ** (max_time - t) for t in timestamps])
#     # Create the decay matrix D = diag(sqrt(w)) (not its inverse)
#     D = np.diag(np.sqrt(w))
#     return D @ K_inv_undecayed @ D

# def apply_decay_with_time(K_undecayed, decay_gamma, X_base_time):
#     timestamps = np.array(X_base_time)
#     max_time = np.max(timestamps)
#     weights = np.array([decay_gamma ** (max_time - t) for t in timestamps])
#     D = np.diag(np.sqrt(weights))
#     return D @ K_undecayed @ D

def apply_decay_with_time(K_undecayed, decay_gamma, X_base_time):
    timestamps = np.array(X_base_time)
    max_time = np.max(timestamps)
    weights = np.array([decay_gamma ** (max_time - t) for t in timestamps])
    D = np.diag(np.sqrt(weights))
    return D @ K_undecayed @ D, weights


# def compute_weighted_GP(X_train, y_train, X_train_time, gamma, X_star, noise, kernel_func, **kernel_args):
#     """
#     Compute the GP predictive mean and standard deviation using weights (decay) on the training data.
#
#     We compute weights for each training point as:
#          w_i = gamma^(t_max - t_i)
#     and form the diagonal matrix W = diag(sqrt(w_i)).
#     Then we weight the kernel matrix and targets:
#          K_w = W @ K @ W + noise * I
#          y_w = W @ y_train
#     and compute the predictive mean as:
#          mu = k_*w^T @ inv(K_w) @ y_w,
#     with k_*w = W @ k_star.
#
#     Args:
#       X_train (np.ndarray): Training inputs of shape (n_train, d).
#       y_train (np.ndarray): Training targets (1-D array).
#       X_train_time (np.ndarray): A 1-D array of timestamps (one per training point).
#       gamma (float): The decay factor, 0 < gamma <= 1.
#       X_star (np.ndarray): Test input(s), shape (n_test, d).
#       noise (float): Noise level to add on the diagonal.
#       kernel_func (callable): The kernel function.
#       **kernel_args: Additional keyword arguments for the kernel.
#
#     Returns:
#       tuple: (mu, std) with predictions (1-D arrays) for the test input(s).
#     """
#     t_max = np.max(X_train_time)
#     # Compute weights for each training point: w_i = gamma^(t_max - t_i)
#     weights = gamma ** (t_max - X_train_time)
#     # Form diagonal weighting matrix
#     W = np.diag(np.sqrt(weights))
#
#     # Compute the training kernel matrix and add noise.
#     K = kernel_func(X_train, X_train, **kernel_args)
#     K_weighted = W @ K @ W + noise * np.eye(len(X_train))
#
#     # Invert the weighted kernel matrix.
#     K_w_inv = np.linalg.inv(K_weighted)
#
#     # Weight the targets.
#     y_weighted = W @ y_train.reshape(-1, 1)
#
#     # Compute k_star for the test input(s).
#     k_star = kernel_func(X_train, X_star, **kernel_args)
#     k_star_weighted = W @ k_star  # weight the covariance between training and test.
#
#     # Predictive mean:
#     mu = k_star_weighted.T @ K_w_inv @ y_weighted
#
#     # Compute k_star_star for the test input(s)
#     k_star_star = kernel_func(X_star, X_star, **kernel_args)
#     # Predictive variance:
#     var = np.diag(k_star_star) - np.sum(k_star_weighted * (K_w_inv @ k_star_weighted), axis=0)
#     std = np.sqrt(np.maximum(var, 0))
#     return mu.ravel(), std


def compute_weighted_GP(X_train, y_train, X_train_time, gamma, X_star, noise, kernel_func, **kernel_args):
    """
    Compute the GP predictive mean and standard deviation using weights (decay) on the training data.
    ...

    Returns:
      tuple: (mu, std) with predictions (1-D arrays) for the test input(s).
    """
    # Ensure that X_train and X_star are at least 2D
    X_train = np.atleast_2d(X_train)
    X_star = np.atleast_2d(X_star)

    t_max = np.max(X_train_time)
    # Compute weights for each training point: w_i = gamma^(t_max - t_i)
    weights = gamma ** (t_max - X_train_time)
    # Form diagonal weighting matrix
    W = np.diag(np.sqrt(weights))

    # Compute the training kernel matrix and add noise.
    K = kernel_func(X_train, X_train, **kernel_args)
    K_weighted = W @ K @ W + noise * np.eye(len(X_train))

    # Invert the weighted kernel matrix.
    K_w_inv = np.linalg.inv(K_weighted)

    # Weight the targets.
    y_weighted = W @ y_train.reshape(-1, 1)

    # Compute k_star for the test input(s).
    k_star = kernel_func(X_train, X_star, **kernel_args)
    k_star_weighted = W @ k_star  # weight the covariance between training and test.

    # Predictive mean:
    mu = k_star_weighted.T @ K_w_inv @ y_weighted

    # Compute k_star_star for the test input(s)
    k_star_star = kernel_func(X_star, X_star, **kernel_args)
    # Predictive variance:
    var = np.diag(k_star_star) - np.sum(k_star_weighted * (K_w_inv @ k_star_weighted), axis=0)
    std = np.sqrt(np.maximum(var, 0))
    return mu.ravel(), std


def get_dataset_path(file_name):
    """
    get the dataset path stored in the project directory.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Real', file_name)
    return path