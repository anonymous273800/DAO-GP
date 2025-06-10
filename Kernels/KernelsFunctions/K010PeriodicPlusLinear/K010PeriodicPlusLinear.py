import numpy as np

def periodic_kernel_1d(X1, X2, length_scale, sigma_f, period):
    X1_first = X1[:, 0:1]
    X2_first = X2[:, 0:1]
    diff = np.abs(X1_first - X2_first.T)
    sin_term = np.sin(np.pi * diff / period) ** 2
    return sigma_f ** 2 * np.exp(-2 * sin_term / (length_scale ** 2))


def linear_kernel(X1, X2, beta_lin):
    if X1.shape[1] > 1:
        X1_rest = X1[:, 1:]
        X2_rest = X2[:, 1:]
        return beta_lin * (X1_rest @ X2_rest.T)
    else:
        # If there are no additional features, return zeros
        return np.zeros((X1.shape[0], X2.shape[0]))


def composite_periodic_plus_linear(X1, X2, per_length, per_sigma, period, beta_lin):
    K_periodic = periodic_kernel_1d(X1, X2, per_length, per_sigma, period)
    K_linear = linear_kernel(X1, X2, beta_lin)
    return K_periodic + K_linear
