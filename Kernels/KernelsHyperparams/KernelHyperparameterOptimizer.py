import numpy as np
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

from scipy import optimize as opt


# --- Rewritten NLML for named dictionary ---
def negative_log_marginal_likelihood(log_params, X, y, kernel_type="rbf", param_names=None, param_shapes=None):
    params = np.exp(log_params)
    grouped = regroup_params(params, param_names, param_shapes)

    if kernel_type == "rbf":
        K = K001RBF.rbf_kernel(X, X, grouped["length_scale"], grouped["sigma_f"]) + grouped["noise"] ** 2 * np.eye(
            len(X))

    elif kernel_type == "periodic":
        K = K002Periodic.periodic_kernel(X, X, grouped["length_scale"], grouped["sigma_f"], grouped["period"]) + \
            grouped["noise"] ** 2 * np.eye(len(X))

    elif kernel_type == "rq":
        K = K003RationalQuadratic.rational_quadratic_kernel(X, X, grouped["length_scale"], grouped["sigma_f"],
                                                            grouped["alpha"]) + grouped["noise"] ** 2 * np.eye(len(X))

    elif kernel_type == "periodic_plus_ard":
        K = K004PeriodicPlusArd.composite_kernel_sum_periodic_plus_ard(
            X, X,
            grouped["per_length"],
            grouped["per_sigma"],
            grouped["period"],
            grouped["rbf_length"],
            grouped["rbf_sigma"],
            grouped["bias"]
        ) + grouped["noise"] ** 2 * np.eye(len(X))

    elif kernel_type == "polynomial_degree3":
        # For the polynomial kernel we add noise linearly (not squared) as in program1.
        K = K005Polynomial.polynomial_kernel(X, X, grouped["coef0"], grouped["scaling_factor"]) + grouped[
            "noise"] * np.eye(len(X))

    elif kernel_type== "log_linear":
        K = K006LogPlusLinear.log_linear_kernel(X, X,grouped["sigma_lin"], grouped["coef0"]) + grouped[
            "noise"] * np.eye(len(X))
    elif kernel_type == "change_point":
        K = K007ChangePoint.change_point_kernel(
            X, X,
            grouped["gamma_poly"],
            grouped["c_poly"],
            grouped["sigma_per"],
            grouped["length_per"],
            grouped["period"],
            grouped["a"]
        ) + grouped["noise"] * np.eye(len(X))
    elif kernel_type == "parabolic_plus_linear":
        # Now period is part of the optimized parameters.
        K = K008ParabolicLinear.parabolic_linear_kernel(
            X, X,
            grouped["gamma_poly"],
            grouped["c_poly"],
            grouped["sigma_per"],
            grouped["length_per"],
            grouped["period"],
            grouped["beta_lin"]
        ) + grouped["noise"] * np.eye(len(X))
    elif kernel_type == "composite_parabolic_periodic_ard_rbf":
        K = K009ParabolicPlusArd.composite_parabolic_periodic_ard_rbf(
            X, X,
            grouped["gamma_poly"],
            grouped["c_poly"],
            grouped["sigma_per"],
            grouped["length_per"],
            grouped["period"],
            grouped["rbf_lengths"],
            grouped["rbf_sigma"]
        ) + grouped["noise"] * np.eye(len(X))
    elif kernel_type == "composite_periodic_plus_linear":
        K = K010PeriodicPlusLinear.composite_periodic_plus_linear(
            X, X,
            grouped["per_length"],
            grouped["per_sigma"],
            grouped["period"],
            grouped["beta_lin"]
        ) + grouped["noise"] * np.eye(len(X))
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    # Add jitter for numerical stability.
    jitter = 1e-6 * np.eye(len(X))
    K += jitter
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return np.inf

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
    nlml = 0.5 * np.dot(y, alpha) + 0.5 * log_det_K + 0.5 * len(X) * np.log(2 * np.pi)
    return nlml


# --- Flatten Parameters ---
def flatten_params(params):
    flat_values, names, bounds, shapes = [], [], [], []
    for p in params:
        val = np.asarray(p["init"])
        bnd = p["bounds"]
        if val.ndim == 0:
            flat_values.append(val.item())
            names.append(p["name"])
            bounds.append((np.log(bnd[0]), np.log(bnd[1])))
            shapes.append(())
        else:
            flat_values.extend(val.tolist())
            names.extend([p["name"]] * val.size)
            bounds.extend([(np.log(bnd[0]), np.log(bnd[1]))] * val.size)
            shapes.extend([val.shape] * val.size)
    return np.log(np.array(flat_values, dtype=float)), bounds, names, shapes


# --- Regroup to Dictionary ---
def regroup_params(flat_values, names, shapes):
    regrouped = {}
    idx = 0
    used = set()
    for i, name in enumerate(names):
        if name in used:
            continue
        used.add(name)
        count = names.count(name)
        shape = shapes[i]
        vals = flat_values[idx:idx + count]
        regrouped[name] = vals[0] if shape == () else np.array(vals).reshape(shape)
        idx += count
    return regrouped


# --- Main Optimizer ---
def optimize_hyperparameters(X, y, kernel_type, params):
    init_log, bounds, names, shapes = flatten_params(params)
    res = opt.minimize(
        fun=negative_log_marginal_likelihood,
        x0=init_log,
        args=(X, y, kernel_type, names, shapes),
        method="L-BFGS-B",
        bounds=bounds
    )
    optimized_flat = np.exp(res.x)
    optimized_dict = regroup_params(optimized_flat, names, shapes)
    print("Optimized hyperparameters:")
    for k, v in optimized_dict.items():
        # Print scalars with 6 decimal places.
        print(f"  {k}: {v if not np.isscalar(v) else f'{v:.6f}'}")
    return optimized_dict
