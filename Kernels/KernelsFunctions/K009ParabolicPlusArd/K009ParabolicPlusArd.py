import numpy as np


def parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period):
    """
    Parabolic-periodic kernel for the first feature:
    k(x₁,x₁') = σ² * [γ·(x₁·x₁') + c]² * exp(-2·sin²(π·|x₁ - x₁'| / p) / ℓ²)
    """
    x1 = X1[:, 0:1]  # First feature
    x2 = X2[:, 0:1]

    # Parabolic component
    parabolic = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2

    # Periodic component
    diff = np.abs(x1 - x2.T)
    periodic = np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))

    return sigma_per ** 2 * parabolic * periodic


def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
    """
    ARD-RBF kernel for the remaining features:
    k(x_rest, x_rest') = σ_f² * exp(-0.5 * sum((x_d - x_d')² / ℓ_d²))
    """
    if X1.shape[1] > 1:
        X1_rest = X1[:, 1:]
        X2_rest = X2[:, 1:]

        diff = X1_rest[:, None, :] - X2_rest[None, :, :]
        sqdist = np.sum((diff ** 2) / (length_scales ** 2), axis=2)

        return sigma_f ** 2 * np.exp(-0.5 * sqdist)

    # If only one feature exists, return zero matrix
    return np.zeros((X1.shape[0], X2.shape[0]))


def composite_parabolic_periodic_ard_rbf(X1, X2,
                               gamma_poly, c_poly, sigma_per, length_per, period,
                               rbf_lengths, rbf_sigma):
    """
    Composite kernel combining:
      - A parabolic-periodic kernel on the first feature
      - An ARD-RBF kernel on the remaining features
    """
    K_par_per = parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly,
                                          sigma_per, length_per, period)
    K_ard = ard_rbf_kernel(X1, X2, rbf_lengths, rbf_sigma)

    return K_par_per + K_ard
