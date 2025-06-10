import numpy as np
def rational_quadratic_kernel(X1, X2, length_scale, sigma_f=1.0, alpha=1.0):
    """
    Computes the Rational Quadratic (RQ) kernel between X1 and X2:

    k(x, x') = sigma_f^2 * (1 + ||x - x'||^2 / (2 * alpha * length_scale^2))^(-alpha)

    Parameters:
    - X1, X2: input arrays of shape (n_samples, n_features)
    - length_scale: scalar controlling length scale
    - sigma_f: signal variance
    - alpha: scale-mixture parameter

    Returns:
    - Kernel matrix of shape (n_samples_X1, n_samples_X2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # Compute squared Euclidean distance
    sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.dot(X1, X2.T)
    )

    # Rational Quadratic kernel formula
    K = sigma_f ** 2 * (1 + sqdist / (2 * alpha * length_scale ** 2)) ** (-alpha)
    return K
