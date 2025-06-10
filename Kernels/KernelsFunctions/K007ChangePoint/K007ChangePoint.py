import numpy as np

# ===== New Change-Point Kernel Implementation =====
def change_point_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period, a):
    """
    Computes the composite change-point kernel.

    For inputs x (first column of X), let:
      s(x) = 1/(1 + exp(a*x))   -- (near 1 for x<<0, near 0 for x>=0)

    Then the kernel is:
      k(x,x') = s(x)s(x')*(gamma_poly * x * x' + c_poly)^2 +
                [1-s(x)][1-s(x')]*sigma_per^2 * exp(-2 sin^2(pi|x-x'|/period) / length_per^2)

    Parameters:
      X1, X2 : np.ndarray with shape (n_samples, d). Only the first feature is used.
      gamma_poly, c_poly : hyperparameters for the polynomial kernel.
      sigma_per, length_per : hyperparameters for the periodic kernel.
      period : the period of the sinusoidal behavior.
      a : steepness of the sigmoid switch.

    Returns:
      Kernel matrix of shape (n_samples1, n_samples2)
    """
    # Use only the first column (assumed to contain the key behavior)
    x1 = X1[:, 0:1]
    x2 = X2[:, 0:1]
    # Compute the sigmoid weights
    s1 = 1.0 / (1.0 + np.exp(a * x1))
    s2 = 1.0 / (1.0 + np.exp(a * x2))
    # Outer products for the weights:
    W_poly = s1 @ s2.T
    W_per = (1 - s1) @ (1 - s2).T
    # Polynomial kernel (degree 2):
    K_poly = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    # Periodic kernel:
    diff = np.abs(x1 - x2.T)
    sin_term = np.sin(np.pi * diff / period) ** 2
    K_per = sigma_per ** 2 * np.exp(-2 * sin_term / (length_per ** 2))
    # Composite kernel is the weighted sum:
    return W_poly * K_poly + W_per * K_per