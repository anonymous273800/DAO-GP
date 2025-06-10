import numpy as np

def parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period):
    x1 = X1[:, 0:1]
    x2 = X2[:, 0:1]
    parabolic = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    diff = np.abs(x1 - x2.T)
    periodic = np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))
    return sigma_per ** 2 * parabolic * periodic

def ard_rbf_kernel(X1, X2, length_scales, sigma_f):
    if X1.shape[1] > 1:
        X1_rest = X1[:, 1:]
        X2_rest = X2[:, 1:]
        diff = X1_rest[:, None, :] - X2_rest[None, :, :]
        sqdist = np.sum((diff ** 2) / (length_scales ** 2), axis=2)
        return sigma_f ** 2 * np.exp(-0.5 * sqdist)
    return np.zeros((X1.shape[0], X2.shape[0]))

def composite_parabolic_periodic_ard_rbf(X1, X2,
                                         gamma_poly=1.0, c_poly=1.0, sigma_per=1.0,
                                         length_per=1.0, period=1.0,
                                         rbf_lengths=None, rbf_sigma=1.0):
    if rbf_lengths is None:
        rbf_lengths = np.ones(X1.shape[1] - 1) if X1.shape[1] > 1 else np.array([])
    K_par_per = parabolic_periodic_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period)
    K_ard = ard_rbf_kernel(X1, X2, rbf_lengths, rbf_sigma)
    return K_par_per + K_ard