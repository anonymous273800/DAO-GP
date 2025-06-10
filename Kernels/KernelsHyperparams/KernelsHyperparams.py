import math

import numpy as np

class RBF:
    """Radial Basis Function (RBF) kernel hyperparameters."""
    params = [
        {"name": "length_scale", "init": 3.0, "bounds": (1e-2, 100)},
        {"name": "sigma_f",      "init": 10.0, "bounds": (1e-3, 100)},
        {"name": "noise",        "init": 0.1, "bounds": (0.1, 10)}
    ]


class RBFPeriodic:
    """RBF-periodic hybrid kernel hyperparameters."""
    params = [
        {"name": "length_scale", "init": 1.0, "bounds": (1e-2, 100)},
        {"name": "sigma_f",      "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "period",       "init": 1.0, "bounds": (1e-2, 100)},
        {"name": "noise",        "init": 0.5, "bounds": (0.1, 10)}
    ]


class RationalQuadratic:
    """Rational Quadratic kernel hyperparameters."""
    params = [
        {"name": "length_scale", "init": 3.0, "bounds": (1e-2, 100)},
        {"name": "sigma_f",      "init": 10.0, "bounds": (1e-3, 100)},
        {"name": "alpha",        "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "noise",        "init": 0.1, "bounds": (0.1, 10)}
    ]


class PolynomialDegree3:
    """Polynomial Degree 3 kernel hyperparameters."""
    params = [
        {"name": "coef0", "init": 3.0, "bounds": (1e-2, 100)},
        {"name": "scaling_factor", "init": 10.0, "bounds": (1e-3, 100)},
        # {"name": "degree", "init": 3.0, "bounds": (3.0, 3.0)},  # Fixed degree
        {"name": "noise", "init": 0.1, "bounds": (0.1, 10)}
    ]


class PeriodicPlusARD:
    @staticmethod
    def get_params(input_dim):
        if input_dim > 1:
            rbf_length_init = np.ones(input_dim - 1)
        else:
            rbf_length_init = 1.0  # or you could return an empty list and adjust the kernel function accordingly
        params = [
            {"name": "per_length", "init": 1.0, "bounds": (1e-2, 100)},
            {"name": "per_sigma", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "period", "init": 1.0, "bounds": (1e-2, 100)},
            # ARD length scales for the non-periodic features (input_dim-1 features)
            {"name": "rbf_length", "init": rbf_length_init, "bounds": (1e-2, 100)},
            {"name": "rbf_sigma", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "bias", "init": 0.0, "bounds": (-10, 10)},  # New bias parameter
            {"name": "noise", "init": 0.5, "bounds": (0.1, 10)}
        ]
        return params


class LogLinear:
    """Log-Linear kernel hyperparameters."""
    params = [
        {"name": "coef0", "init": 3.0, "bounds": (1e-2, 100)},
        {"name": "sigma_lin", "init": 10.0, "bounds": (1e-3, 100)},
        {"name": "noise", "init": 0.1, "bounds": (0.1, 10)}
    ]


class ChangePoint:
    params = [
        {"name": "gamma_poly", "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "c_poly", "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "sigma_per", "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "length_per", "init": 1.0, "bounds": (1e-3, 100)},
        {"name": "period", "init": 1.2566, "bounds": (0.1, 10)},
        {"name": "a", "init": 10.0, "bounds": (0.1, 50)},
        {"name": "noise", "init": 0.1, "bounds": (0.1, 10)}
    ]


class ParabolicPlusLinear:
    @staticmethod
    def get_params(stretch_factor):
        params = [
            {"name": "gamma_poly", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "c_poly", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "sigma_per", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "length_per", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "period", "init": 2 * math.pi / stretch_factor, "bounds": (2 * math.pi / stretch_factor, 2 * math.pi / stretch_factor)},
            {"name": "beta_lin", "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "noise", "init": 0.1, "bounds": (0.1, 10)}
        ]
        return  params

class CompositeParabolicPeriodicArdRbf:
    """Parabolic + Periodic (first feature) and ARD-RBF (remaining features) kernel hyperparameters."""
    @staticmethod
    def get_params(input_dim):
        if input_dim > 1:
            rbf_length_init = np.ones(input_dim - 1)
        else:
            rbf_length_init = np.array([1.0])  # Keep it as array for consistency

        params = [
            {"name": "gamma_poly",  "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "c_poly",      "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "sigma_per",   "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "length_per",  "init": 1.0, "bounds": (1e-2, 100)},
            {"name": "period",      "init": 1.0, "bounds": (1e-2, 100)},
            {"name": "rbf_lengths", "init": rbf_length_init, "bounds": (1e-2, 100)},
            {"name": "rbf_sigma",   "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "noise",       "init": 0.5, "bounds": (0.1, 10)}
        ]
        return params


class CompositePeriodicPlusLinear:
    """Periodic (on first feature) + Linear (on remaining features) kernel hyperparameters."""
    @staticmethod
    def get_params(input_dim):
        params = [
            {"name": "per_length", "init": 1.0, "bounds": (1e-2, 100)},
            {"name": "per_sigma",  "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "period",     "init": 1.0, "bounds": (1e-2, 100)},
            {"name": "beta_lin",   "init": 1.0, "bounds": (1e-3, 100)},
            {"name": "noise",      "init": 0.5, "bounds": (0.1, 10)}
        ]
        return params

class KernelsHyperparams:
    """Helper class to map kernel types to their hyperparameter configs."""

    @staticmethod
    def get_params(kernel_type, input_dim=None, stretch_factor=None):
        kernel_type = kernel_type.lower()
        if kernel_type == "rbf":
            return RBF.params
        elif kernel_type == "periodic":
            return RBFPeriodic.params
        elif kernel_type == "rq":
            return RationalQuadratic.params
        elif kernel_type == "periodic_plus_ard":
            if input_dim is None:
                raise ValueError("input_dim must be provided for periodic_plus_ard")
            return PeriodicPlusARD.get_params(input_dim)
        elif kernel_type == "polynomial_degree3":
            return PolynomialDegree3.params
        elif kernel_type == "log_linear":
            return LogLinear.params
        elif kernel_type == "change_point":
            return ChangePoint.params
        elif kernel_type == "parabolic_plus_linear":
            return ParabolicPlusLinear.get_params(stretch_factor)
        elif kernel_type == "composite_parabolic_periodic_ard_rbf":
            if input_dim is None:
                raise ValueError("input_dim must be provided for parabolic_plus_ard")
            return CompositeParabolicPeriodicArdRbf.get_params(input_dim)
        elif kernel_type == "composite_periodic_plus_linear":
            if input_dim is None:
                raise ValueError("input_dim must be provided for parabolic_plus_linear")
            return CompositePeriodicPlusLinear.get_params(input_dim)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    @staticmethod
    def get_kernel_function(kernel_type):
        from Kernels.KernelsFunctions.K001RBF import K001RBF
        from Kernels.KernelsFunctions.K002Periodic import K002Periodic
        from Kernels.KernelsFunctions.K003RationalQuadratic import K003RationalQuadratic
        from Kernels.KernelsFunctions.K004PeriodicPlusArd import K004PeriodicPlusArd
        from Kernels.KernelsFunctions.K005Polynomial import K005Polynomial
        from Kernels.KernelsFunctions.K006LogPlusLinear import K006LogPlusLinear
        from Kernels.KernelsFunctions.K007ChangePoint import K007ChangePoint
        from Kernels.KernelsFunctions.K008ParabolicLinear import K008ParabolicLinear
        from Kernels.KernelsFunctions.K009ParabolicPlusArd import K009ParabolicPlusArd
        from Kernels.KernelsFunctions.K010PeriodicPlusLinear import K010PeriodicPlusLinear
        kernel_type = kernel_type.lower()

        if kernel_type == "rbf":
            return K001RBF.rbf_kernel_ard
        elif kernel_type == "periodic":
            return K002Periodic.periodic_kernel
        elif kernel_type == "rq":
            return K003RationalQuadratic.rational_quadratic_kernel
        elif kernel_type == "periodic_plus_ard":
            return K004PeriodicPlusArd.composite_kernel_sum_periodic_plus_ard
        if kernel_type == "polynomial_degree3":
            return K005Polynomial.polynomial_kernel
        elif kernel_type == "log_linear":
            return K006LogPlusLinear.log_linear_kernel
        elif kernel_type == "change_point":
            return K007ChangePoint.change_point_kernel
        elif kernel_type == "parabolic_plus_linear":
            return K008ParabolicLinear.parabolic_linear_kernel
        elif kernel_type == "composite_parabolic_periodic_ard_rbf":
            return K009ParabolicPlusArd.composite_parabolic_periodic_ard_rbf
        elif kernel_type == "composite_periodic_plus_linear":
            return K010PeriodicPlusLinear.composite_periodic_plus_linear
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
