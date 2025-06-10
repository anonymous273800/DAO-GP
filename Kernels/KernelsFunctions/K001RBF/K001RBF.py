import numpy as np

def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    RBF (squared exponential) kernel:
      K[i, j] = sigma_f^2 * exp(-||X1[i]-X2[j]||^2/(2*length_scale^2))
    """
    sqdist = (np.sum(X1 ** 2, axis=1).reshape(-1, 1) +
              np.sum(X2 ** 2, axis=1) -
              2 * X1 @ X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / (length_scale ** 2) * sqdist)


def rbf_kernel_ard(X1, X2, length_scale, sigma_f=1.0):
    # If length_scale is scalar, replicate it for all dimensions.
    if np.isscalar(length_scale):
        length_scale = np.array([length_scale] * X1.shape[1])
    X1_scaled = X1 / length_scale
    X2_scaled = X2 / length_scale
    sqdist = (np.sum(X1_scaled ** 2, axis=1).reshape(-1, 1) +
              np.sum(X2_scaled ** 2, axis=1) - 2 * np.dot(X1_scaled, X2_scaled.T))
    return sigma_f ** 2 * np.exp(-0.5 * sqdist)


