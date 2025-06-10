import GPy
import numpy as np

class CompositeParabolicPeriodicARDRBF(GPy.kern.Kern):
    def __init__(self, input_dim, active_dims=None, name='CompositeParabolicPeriodicARDRBF'):
        super().__init__(input_dim, active_dims, name)

        # Parameters for parabolic + periodic component (on dim 0)
        self.gamma_poly = GPy.core.Param('gamma_poly', 1.0, GPy.core.parameterization.transformations.Logexp())
        self.c_poly = GPy.core.Param('c_poly', 1.0, GPy.core.parameterization.transformations.Logexp())
        self.sigma_per = GPy.core.Param('sigma_per', 1.0, GPy.core.parameterization.transformations.Logexp())
        self.length_per = GPy.core.Param('length_per', 1.0, GPy.core.parameterization.transformations.Logexp())
        self.period = GPy.core.Param('period', 1.0, GPy.core.parameterization.transformations.Logexp())

        # ARD RBF parameters for remaining dims if more than one feature
        if input_dim > 1:
            self.rbf_lengths = GPy.core.Param('rbf_lengths', np.ones(input_dim - 1),
                                            GPy.core.parameterization.transformations.Logexp())
            self.rbf_sigma = GPy.core.Param('rbf_sigma', 1.0,
                                           GPy.core.parameterization.transformations.Logexp())
            self.link_parameters(self.rbf_lengths, self.rbf_sigma)

        self.link_parameters(self.gamma_poly, self.c_poly,
                           self.sigma_per, self.length_per, self.period)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # 0th dim parabolic + periodic kernel
        x1 = X[:, 0:1]
        x2 = X2[:, 0:1]
        dot_poly = self.gamma_poly * (x1 @ x2.T) + self.c_poly
        K_poly = dot_poly ** 2

        diff = np.abs(x1 - x2.T)
        K_per = np.exp(-2 * (np.sin(np.pi * diff / self.period) ** 2) / (self.length_per ** 2))

        K_nonlin = self.sigma_per ** 2 * K_poly * K_per

        # RBF kernel on remaining features
        if X.shape[1] > 1:
            X_rem = X[:, 1:]
            X2_rem = X2[:, 1:]
            scaled_X = X_rem / self.rbf_lengths
            scaled_X2 = X2_rem / self.rbf_lengths
            sq_dist = np.sum((scaled_X[:, None, :] - scaled_X2[None, :, :]) ** 2, axis=2)
            K_rbf = self.rbf_sigma ** 2 * np.exp(-0.5 * sq_dist)
        else:
            K_rbf = 0

        return K_nonlin + K_rbf

    def Kdiag(self, X):
        x1 = X[:, 0:1]
        dot_poly = self.gamma_poly * (x1 * x1) + self.c_poly
        K_poly_diag = dot_poly ** 2
        K_per_diag = np.ones_like(K_poly_diag)
        K_nonlin_diag = self.sigma_per ** 2 * K_poly_diag.flatten() * K_per_diag.flatten()

        if X.shape[1] > 1:
            K_rbf_diag = self.rbf_sigma ** 2 * np.ones(X.shape[0])
        else:
            K_rbf_diag = 0

        return K_nonlin_diag + K_rbf_diag