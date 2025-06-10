import numpy as np


# ==== Updated Log-Linear Kernel (using only the first feature) ====
def log_linear_kernel(X1, X2, sigma_lin, coef0):
    """
    Computes a linear kernel on the log-transformed first feature.

    k(x,x') = sigma_lin^2 * dot( log(x_first), log(x'_first) ) + coef0

    Only the first column (feature) of X1 and X2 is used for the logarithmic transformation.
    """
    # Only transform the first column (assumed to be positive)
    logX1 = np.log(X1[:, 0:1])
    logX2 = np.log(X2[:, 0:1])
    return sigma_lin ** 2 * np.dot(logX1, logX2.T) + coef0