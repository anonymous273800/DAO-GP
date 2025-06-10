import GPy
import numpy as np

class ParabolicPeriodicLinear(GPy.kern.Kern):
    def __init__(self, input_dim,gamma_poly, c_poly, sigma_per, length_per, period, active_dims=None, name='ParabolicPeriodicLinear'):
        super(ParabolicPeriodicLinear, self).__init__(input_dim, active_dims, name)

        # Parameters for parabolic + periodic component (on dim 0)
        self.gamma_poly = GPy.core.Param('gamma_poly', gamma_poly)
        self.c_poly = GPy.core.Param('c_poly', c_poly)
        self.sigma_per = GPy.core.Param('sigma_per', sigma_per)
        self.length_per = GPy.core.Param('length_per', length_per)
        self.period = GPy.core.Param('period', period)

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




import GPy
import numpy as np

class CompositeParabolicPeriodicARDRBF(GPy.kern.Kern):
    """
    Composite kernel that applies:
      1. A parabolic‐periodic kernel on the first input dimension
      2. An ARD‐RBF kernel on all remaining dimensions
    """
    def __init__(self, input_dim, gamma_poly, c_poly, sigma_per,length_per, period,  active_dims=None, name='CompositeParabolicPeriodicARDRBF'):
        super(CompositeParabolicPeriodicARDRBF, self).__init__(input_dim, active_dims, name)

        # Parameters for the parabolic‐periodic component (on dim 0)
        # self.gamma_poly = GPy.core.Param('gamma_poly', 1.0)
        # self.c_poly     = GPy.core.Param('c_poly',     1.0)
        # self.sigma_per  = GPy.core.Param('sigma_per',  1.0)
        # self.length_per = GPy.core.Param('length_per', 1.0)
        # self.period     = GPy.core.Param('period',     1.0)

        self.gamma_poly = GPy.core.Param('gamma_poly', gamma_poly)
        self.c_poly = GPy.core.Param('c_poly', c_poly)
        self.sigma_per = GPy.core.Param('sigma_per', sigma_per)
        self.length_per = GPy.core.Param('length_per', length_per)
        self.period = GPy.core.Param('period', period)

        # Parameters for the ARD‐RBF component (on dims 1..end)
        if input_dim > 1:
            # One length‐scale per remaining dimension
            self.rbf_lengths = GPy.core.Param('rbf_lengths', np.ones(input_dim - 1))
        else:
            # No “remaining” dims → keep a zero‐length array
            self.rbf_lengths = GPy.core.Param('rbf_lengths', np.array([]))

        # Overall output variance for the RBF part
        self.rbf_sigma = GPy.core.Param('rbf_sigma', 1.0)

        # Link all parameters
        self.link_parameters(self.gamma_poly, self.c_poly,
                             self.sigma_per, self.length_per, self.period,
                             self.rbf_lengths, self.rbf_sigma)

    def K(self, X, X2=None):
        """
        Full kernel matrix K(X, X2) = K_parabolic-periodic(X, X2) + K_ARD-RBF(X, X2).
        """
        if X2 is None:
            X2 = X

        # --- 1) Parabolic‐periodic on the first feature ---
        x1  = X[:,  0:1]   # shape (N, 1)
        x2  = X2[:, 0:1]   # shape (M, 1)
        # Parabolic part: [γ·(x₁·x₁') + c]²
        dot_poly = self.gamma_poly * (x1 @ x2.T) + self.c_poly        # shape (N, M)
        K_poly   = dot_poly ** 2                                       # shape (N, M)

        # Periodic part: exp(–2·sin²(π·|x₁ – x₁'| / p) / ℓ²)
        diff      = np.abs(x1 - x2.T)                                  # shape (N, M)
        sin_term  = np.sin(np.pi * diff / self.period)                 # shape (N, M)
        K_per     = (self.sigma_per ** 2) * np.exp(-2 * (sin_term ** 2) / (self.length_per ** 2))
        K_par_per = K_poly * K_per                                      # shape (N, M)

        # --- 2) ARD‐RBF on remaining features (dims 1…) ---
        if X.shape[1] > 1:
            X_rest  = X[:,  1:]   # shape (N, D–1)
            X2_rest = X2[:, 1:]   # shape (M, D–1)

            # Compute squared distances scaled by the length‐scales:
            #    sum_d [ (X_rest[n,d] – X2_rest[m,d])² / ℓ_d² ]
            # ℓ_d = self.rbf_lengths[d]
            # We broadcast so that “diff_rest” has shape (N, M, D–1)
            diff_rest = (X_rest[:, None, :] - X2_rest[None, :, :]) / self.rbf_lengths
            sqdist    = np.sum(diff_rest ** 2, axis=2)  # shape (N, M)
            K_ard     = (self.rbf_sigma ** 2) * np.exp(-0.5 * sqdist)
        else:
            # If no “remaining” dims, ARD‐RBF contributes all zeros
            K_ard = np.zeros((X.shape[0], X2.shape[0]))

        return K_par_per + K_ard

    def Kdiag(self, X):
        """
        Diagonal of the composite kernel: for each xᵢ,
          K_par_per(xᵢ, xᵢ) + K_ard(xᵢ, xᵢ).
        """
        # Parabolic‐periodic diagonal on the first feature
        x1 = X[:, 0:1]                                  # shape (N, 1)
        dot_poly_diag = self.gamma_poly * (x1 * x1) + self.c_poly  # shape (N, 1)
        K_poly_diag   = dot_poly_diag ** 2                      # shape (N, 1)
        # Periodic on the diagonal is just σ_per² (since diff=0 → sin(0)=0 → exp(0)=1)
        K_per_diag    = (self.sigma_per ** 2) * np.ones_like(K_poly_diag)
        K_par_per_diag = (K_poly_diag.flatten() * K_per_diag.flatten())

        # ARD‐RBF diagonal: for each xᵢ in dims 1..end,
        #   K_ard(xᵢ, xᵢ) = σ_f² * exp(–0.5·0) = σ_f² (if at least one remaining feature)
        if X.shape[1] > 1:
            K_ard_diag = (self.rbf_sigma ** 2) * np.ones(X.shape[0])
        else:
            K_ard_diag = np.zeros(X.shape[0])

        return K_par_per_diag + K_ard_diag
