import numpy as np

def periodic_kernel(X1, X2, length_scale, sigma_f=1.0, period=1.0):
    """
    Computes the periodic kernel between X1 and X2:
    k(x, x') = sigma_f^2 * exp(-2 * sin^2(pi * ||x - x'|| / period) / length_scale^2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    dist = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
    sin_term = np.sin(np.pi * dist / period) ** 2
    return sigma_f ** 2 * np.exp(-2 * sin_term / length_scale ** 2)
