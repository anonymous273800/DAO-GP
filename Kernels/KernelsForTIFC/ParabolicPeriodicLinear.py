import GPy
import numpy as np

class ParabolicPeriodicLinear(GPy.kern.Kern):
    def __init__(self, input_dim, active_dims=None, name='ParabolicPeriodicLinear'):
        super(ParabolicPeriodicLinear, self).__init__(input_dim, active_dims, name)

        # Parameters for parabolic + periodic component (on dim 0)
        self.gamma_poly = GPy.core.Param('gamma_poly', 1.0)
        self.c_poly = GPy.core.Param('c_poly', 1.0)
        self.sigma_per = GPy.core.Param('sigma_per', 1.0)
        self.length_per = GPy.core.Param('length_per', 1.0)
        self.period = GPy.core.Param('period', 1.0)

        # Parameter for linear component (dims 1+)
        self.beta_lin = GPy.core.Param('beta_lin', 1.0)

        self.link_parameters(self.gamma_poly, self.c_poly, self.sigma_per,
                             self.length_per, self.period, self.beta_lin)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # First feature only
        x1 = X[:, 0:1]
        x2 = X2[:, 0:1]
        dot_poly = self.gamma_poly * (x1 @ x2.T) + self.c_poly
        K_poly = dot_poly ** 2

        diff = np.abs(x1 - x2.T)
        K_per = self.sigma_per ** 2 * np.exp(-2 * (np.sin(np.pi * diff / self.period) ** 2) / (self.length_per ** 2))
        K_nonlin = K_poly * K_per

        # Remaining features (if any)
        if X.shape[1] > 1:
            X_rem = X[:, 1:]
            X2_rem = X2[:, 1:]
            K_lin = self.beta_lin ** 2 * (X_rem @ X2_rem.T)
        else:
            K_lin = 0

        return K_nonlin + K_lin

    def Kdiag(self, X):
        # Diagonal of the kernel matrix
        x1 = X[:, 0:1]
        dot_poly = self.gamma_poly * (x1 * x1) + self.c_poly
        K_poly_diag = dot_poly ** 2
        K_per_diag = self.sigma_per ** 2 * np.ones_like(K_poly_diag)
        K_nonlin_diag = K_poly_diag.flatten() * K_per_diag.flatten()

        if X.shape[1] > 1:
            K_lin_diag = self.beta_lin ** 2 * np.sum(X[:, 1:] ** 2, axis=1)
        else:
            K_lin_diag = 0

        return K_nonlin_diag + K_lin_diag