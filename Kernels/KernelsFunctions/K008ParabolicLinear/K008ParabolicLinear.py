import numpy as np

# ===== New Composite Kernel: Parabolic + Linear =====
def parabolic_linear_kernel(X1, X2, gamma_poly, c_poly, sigma_per, length_per, period, beta_lin):
    """
    Composite kernel for DS007_ParabolicWave.

    For the first feature (assumed to carry the nonlinear parabolic wave):
      k_nonlin(x,x') = [gamma_poly * x*x' + c_poly]^2 *
                        sigma_per^2 * exp(-2*sin^2(pi*|x-x'|/period)/(length_per^2))

    For remaining features (if any), we use a linear kernel:
      k_lin(x,x') = beta_lin^2 * (x_{2:} dot x'_{2:})

    The composite kernel is defined as:
      k(x,x') = k_nonlin(x_1,x'_1) + k_lin(x_{2:},x'_{2:})
    """
    # Use only the first column for the nonlinear part.
    x1 = X1[:, 0:1]
    x2 = X2[:, 0:1]
    K_nonlin = (gamma_poly * (x1 @ x2.T) + c_poly) ** 2
    # Multiply by the periodic factor:
    diff = np.abs(x1 - x2.T)
    K_per = sigma_per ** 2 * np.exp(-2 * (np.sin(np.pi * diff / period) ** 2) / (length_per ** 2))
    K1 = K_nonlin * K_per

    # For additional features (if any), use a linear kernel.
    if X1.shape[1] > 1 and X2.shape[1] > 1:
        K_lin = beta_lin ** 2 * (X1[:, 1:] @ X2[:, 1:].T)
    else:
        K_lin = 0

    return K1 + K_lin