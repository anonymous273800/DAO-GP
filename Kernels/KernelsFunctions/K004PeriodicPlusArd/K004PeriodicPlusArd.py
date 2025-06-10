import numpy as np

def periodic_kernel_1d(X1, X2, length_scale, sigma_f, period):
    # Use only the first feature (assumed to contain the periodic signal)
    X1_first = X1[:, 0:1]
    X2_first = X2[:, 0:1]
    diff = np.abs(X1_first - X2_first.T)
    sin_term = np.sin(np.pi * diff / period) ** 2
    return sigma_f ** 2 * np.exp(-2 * sin_term / (length_scale ** 2))


def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
    # Apply ARD on the remaining features (if any)
    if X1.shape[1] > 1:
        # Use features 1:end for the ARD kernel
        X1_rest = X1[:, 1:]
        X2_rest = X2[:, 1:]
        # Compute the squared distance for each dimension scaled by its own length scale.
        diff = X1_rest[:, None, :] - X2_rest[None, :, :]
        # (x_d - x'_d)^2 / (length_scale_d)^2
        sqdist = np.sum(diff**2 / (length_scales**2), axis=2)
        return sigma_f ** 2 * np.exp(-0.5 * sqdist)
    else:
        # If there is no additional dimension, return zeros.
        return np.zeros((X1.shape[0], X2.shape[0]))


# def composite_kernel_sum_periodic_plus_ard(X1, X2, per_length, per_sigma, period, rbf_length, rbf_sigma):
#     return periodic_kernel_1d(X1, X2, per_length, per_sigma, period) + \
#            ard_rbf_kernel(X1, X2, rbf_length, rbf_sigma)

def composite_kernel_sum_periodic_plus_ard(X1, X2, per_length, per_sigma, period, rbf_length, rbf_sigma, bias):
    K_periodic = periodic_kernel_1d(X1, X2, per_length, per_sigma, period)
    K_ard = ard_rbf_kernel(X1, X2, rbf_length, rbf_sigma)
    # K_bias = K000ConstantKernel.constant_kernel(X1,X2,bias)
    return K_periodic + K_ard #+ K_bias

# def composite_kernel_sum_periodic_plus_ard_with_bias(X1, X2, per_length, per_sigma, period, rbf_length, rbf_sigma, bias):
#     K_periodic = periodic_kernel_1d(X1, X2, per_length, per_sigma, period)
#     K_ard = ard_rbf_kernel(X1, X2, rbf_length, rbf_sigma)
#     K_bias = bias * np.ones((X1.shape[0], X2.shape[0]))
#     return K_periodic + K_ard + K_bias


# def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
#     # Ensure the inputs are 2D arrays
#     X1 = np.atleast_2d(X1)
#     X2 = np.atleast_2d(X2)
#
#     # Scale the inputs appropriately using the provided length_scales
#     X1_scaled = X1 / length_scales
#     X2_scaled = X2 / length_scales
#
#     # Compute the squared distance matrix
#     sqdist = (np.sum(X1_scaled ** 2, axis=1).reshape(-1, 1) +
#               np.sum(X2_scaled ** 2, axis=1) -
#               2 * np.dot(X1_scaled, X2_scaled.T))
#
#     # Return the kernel matrix
#     return sigma_f ** 2 * np.exp(-0.5 * sqdist)
