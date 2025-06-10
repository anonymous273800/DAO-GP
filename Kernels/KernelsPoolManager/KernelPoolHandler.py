from sklearn.metrics import mean_squared_error, r2_score
from Kernels.KernelsPoolManager import KernelsPool
from Kernels.KernelsHyperparams import KernelsHyperparams
from Kernels.KernelsHyperparams import KernelHyperparameterOptimizer
from Utils import Util
from PerformanceRecorder.MBPerformance import MBPerformance
from PerformanceRecorder.MBPerformanceManager import MBPerformanceManager
import numpy as np

"""
We first choose the kernel with the highest R². Then we check if its MSE is within tol of the overall best MSE. If so, we keep that kernel. Otherwise, we look for any candidate kernel that is (within tolerance) nearly best in both R² and MSE. If none are found, we default back to the one with the highest R².
"""


# def pick_best_kernel(INITIAL_KERNEL, X_win, y_win, X_val, y_val, kernels_list, input_dim, stretch_factor=None, decay_gamma=0.99,tol=1e-6):
#     results = {}
#     jitter = 1e-6 * np.eye(len(y_win))
#
#     for kernel_type in kernels_list:
#
#         print("********* KENEL ON PROCESS: ", kernel_type, "**********")
#         # Get the hyperparameter configuration.
#         params = KernelsHyperparams.KernelsHyperparams.get_params(kernel_type, input_dim=input_dim, stretch_factor=stretch_factor)
#         kernel_func = KernelsHyperparams.KernelsHyperparams.get_kernel_function(kernel_type)
#
#         # Optimize hyperparameters on the training set (X_win, y_win).
#         opt_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_win, y_win, kernel_type, params)
#
#         noise = opt_dict["noise"]
#         kernel_args = {k: v for k, v in opt_dict.items() if k != "noise"}
#         # Compute predictions on the validation set.
#         K_base = kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(y_win)) + jitter
#         K_inv = np.linalg.inv(K_base)
#
#         mu_val, _ = Util.predict_with_decay(X_window=X_win,y_window=y_win,K_inv_undecayed=K_inv,X_star=X_val,noise=noise,decay_gamma=decay_gamma, decay_option=False,kernel_func=kernel_func,**kernel_args)
#
#         if np.isnan(mu_val).any():
#             curr_r2 = -np.inf
#             curr_mse = np.inf
#         else:
#             curr_r2 = r2_score(y_val, mu_val)
#             curr_mse = mean_squared_error(y_val, mu_val)
#
#         print(f"Kernel {kernel_type}: R² = {curr_r2:.6f}, MSE = {curr_mse:.6f}")
#         results[kernel_type] = (curr_r2, curr_mse, opt_dict, kernel_func, kernel_args)
#
#     # Determine overall best metrics.
#     best_r2_overall = max(v[0] for v in results.values())
#     best_mse_overall = min(v[1] for v in results.values())
#
#     # First choose the kernel with the highest R².
#     chosen_kernel = max(results.items(), key=lambda item: item[1][0])[0]
#     best_r2, best_mse, best_opt_dict, best_kernel_func, best_kernel_args = results[chosen_kernel]
#
#     # If the chosen kernel’s MSE is not nearly the lowest overall, try to find a candidate
#     # that is nearly best in both R² and MSE.
#     if abs(best_mse - best_mse_overall) >= tol:
#         candidates = [
#             kernel for kernel, (r2_val, mse_val, _, _, _) in results.items()
#             if abs(r2_val - best_r2_overall) < tol and abs(mse_val - best_mse_overall) < tol
#         ]
#         if candidates:
#             chosen_kernel = candidates[0]
#             best_r2, best_mse, best_opt_dict, best_kernel_func, best_kernel_args = results[chosen_kernel]
#
#     print("##################################################################")
#     print(f"**Best kernel selected: {chosen_kernel} with R² = {best_r2:.6f} and MSE = {best_mse:.6f}")
#     print("** Kernel Switch - from ", INITIAL_KERNEL, "TO: ", chosen_kernel)
#
#     # Also return the default hyperparameter configuration for the chosen kernel.
#     best_kernel_params = KernelsHyperparams.KernelsHyperparams.get_params(chosen_kernel, input_dim=input_dim,
#                                                                           stretch_factor=stretch_factor)
#     print("##################################################################")
#     return chosen_kernel, best_kernel_params, best_opt_dict, best_kernel_func, best_kernel_args, best_r2, best_mse



# def pick_best_kernel_drift(INITIAL_KERNEL, X_win, y_win, kernels_list, input_dim, stretch_factor=None, decay_gamma=0.99, tol=1e-6):
#     results = {}
#     jitter = 1e-6 * np.eye(len(y_win))
#
#     for kernel_type in kernels_list:
#         print("********* KENEL ON PROCESS: ", kernel_type, "**********")
#         # Get the hyperparameter configuration.
#         params = KernelsHyperparams.KernelsHyperparams.get_params(kernel_type, input_dim=input_dim, stretch_factor=stretch_factor)
#
#         # Optimize hyperparameters on the training set (X_win, y_win).
#         opt_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_win, y_win, kernel_type, params)
#         kernel_func = KernelsHyperparams.KernelsHyperparams.get_kernel_function(kernel_type)
#         noise = opt_dict["noise"]
#         kernel_args = {k: v for k, v in opt_dict.items() if k != "noise"}
#         # Compute predictions on the validation set.
#         K_base = kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(y_win)) + jitter
#         K_inv = np.linalg.inv(K_base)
#
#         mu_trn, _ = Util.predict_with_decay(X_window=X_win,y_window=y_win,K_inv_undecayed=K_inv,X_star=X_win,noise=noise,decay_gamma=decay_gamma,decay_option=False,kernel_func=kernel_func,**kernel_args)
#
#         if np.isnan(mu_trn).any():
#             curr_r2 = -np.inf
#             curr_mse = np.inf
#         else:
#             curr_r2 = r2_score(y_win, mu_trn)
#             curr_mse = mean_squared_error(y_win, mu_trn)
#
#         KernelsPool.kernels_pool[kernel_type]["visited"] = True
#         print(f"Kernel {kernel_type}: R² = {curr_r2:.6f}, MSE = {curr_mse:.6f}")
#         results[kernel_type] = (curr_r2, curr_mse, opt_dict, kernel_func, kernel_args)
#
#     # Determine overall best metrics.
#     best_r2_overall = max(v[0] for v in results.values())
#     best_mse_overall = min(v[1] for v in results.values())
#
#     # First choose the kernel with the highest R².
#     chosen_kernel = max(results.items(), key=lambda item: item[1][0])[0]
#     best_r2, best_mse, best_opt_dict, best_kernel_func, best_kernel_args = results[chosen_kernel]
#
#     # If the chosen kernel’s MSE is not nearly the lowest overall, try to find a candidate
#     # that is nearly best in both R² and MSE.
#     if abs(best_mse - best_mse_overall) >= tol:
#         candidates = [
#             kernel for kernel, (r2_val, mse_val, _, _, _) in results.items()
#             if abs(r2_val - best_r2_overall) < tol and abs(mse_val - best_mse_overall) < tol
#         ]
#         if candidates:
#             chosen_kernel = candidates[0]
#             best_r2, best_mse, best_opt_dict, best_kernel_func, best_kernel_args = results[chosen_kernel]
#
#     print("##################################################################")
#     print(f"**Best kernel selected: {chosen_kernel} with R² = {best_r2:.6f} and MSE = {best_mse:.6f}")
#     print("** Kernel Switch - from ", INITIAL_KERNEL, "TO: ", chosen_kernel)
#
#     # Also return the default hyperparameter configuration for the chosen kernel.
#     best_kernel_params = KernelsHyperparams.KernelsHyperparams.get_params(chosen_kernel, input_dim=input_dim,stretch_factor=stretch_factor)
#     print("##################################################################")
#     return chosen_kernel, best_kernel_params, best_opt_dict, best_kernel_func, best_kernel_args, best_r2, best_mse

def pick_best_kernel(INITIAL_KERNEL, X_base_tr, y_base_tr, X_base_vl, y_base_vl,
                     KERNEL_POOL, n_features, STRETCH_FACTOR, kpi, tol=1e-6):
    results = {}
    jitter = 1e-6 * np.eye(len(y_base_tr))


    for kernel_type in KERNEL_POOL:
        print("********* KERNEL IN PROCESS:", kernel_type, "**********")

        params = KernelsHyperparams.KernelsHyperparams.get_params(kernel_type, input_dim=n_features, stretch_factor=STRETCH_FACTOR)
        kernel_func = KernelsHyperparams.KernelsHyperparams.get_kernel_function(kernel_type)

        opt_dict = KernelHyperparameterOptimizer.optimize_hyperparameters(X_base_tr, y_base_tr, kernel_type, params)

        noise = opt_dict["noise"]
        kernel_args = {k: v for k, v in opt_dict.items() if k != "noise"}

        K_base = kernel_func(X_base_tr, X_base_tr, **kernel_args) + noise * np.eye(len(y_base_tr)) + jitter
        K_inv = np.linalg.inv(K_base)

        mu_val, _ = Util.computeGP(
            X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv,
            X_star=X_base_vl, noise=noise,
            kernel_func=kernel_func, **kernel_args)
        mu_trn, _ = Util.computeGP(
            X_window=X_base_tr, y_window=y_base_tr, K_inv=K_inv,
            X_star=X_base_tr, noise=noise,
            kernel_func=kernel_func, **kernel_args)

        if np.isnan(mu_val).any():
            curr_r2 = -np.inf
            curr_mse = np.inf
        else:
            curr_r2 = r2_score(y_base_vl, mu_val)
            curr_mse = mean_squared_error(y_base_vl, mu_val)

        if np.isnan(mu_trn).any():
            curr_r2_tr = -np.inf
            curr_mse_tr = np.inf
        else:
            curr_r2_tr = r2_score(y_base_tr, mu_trn)
            curr_mse_tr = mean_squared_error(y_base_tr, mu_trn)

        print(f"Kernel VALIDATION {kernel_type}: R² = {curr_r2:.6f}, MSE = {curr_mse:.6f}")

        results[kernel_type] = {
            'R2': curr_r2,
            'MSE': curr_mse,
            'R2_tr': curr_r2_tr,
            'MSE_tr': curr_mse_tr,
            'opt_dict': opt_dict,
            'kernel_func': kernel_func,
            'kernel_args': kernel_args
        }

    # Select kernel based on KPI
    print("****************************************************** kpi", kpi)
    if kpi == 'R2':
        chosen_kernel = max(results.items(), key=lambda item: item[1]['R2'])[0]
    elif kpi == 'MSE':
        chosen_kernel = min(results.items(), key=lambda item: item[1]['MSE'])[0]
    else:
        raise ValueError(f"Unsupported KPI: {kpi}. Choose 'R2' or 'MSE'.")

    acceptable_diff_threshold = 0.005
    initial_kpi_value = results[INITIAL_KERNEL][kpi]
    chosen_kpi_value = results[chosen_kernel][kpi]


    if ((kpi == 'R2' and initial_kpi_value >= chosen_kpi_value - acceptable_diff_threshold) or
            (kpi == 'MSE' and initial_kpi_value <= chosen_kpi_value + acceptable_diff_threshold)):
        print( f"** Preference: Keeping INITIAL_KERNEL ({INITIAL_KERNEL}) within threshold ({acceptable_diff_threshold})")
        chosen_kernel = INITIAL_KERNEL

    selected = results[chosen_kernel]
    best_r2 = selected['R2']
    best_mse = selected['MSE']
    best_r2_tr = selected['R2_tr']
    best_mse_tr = selected['MSE_tr']
    best_opt_dict = selected['opt_dict']
    best_kernel_func = selected['kernel_func']
    best_kernel_args = selected['kernel_args']
    best_kernel_params = KernelsHyperparams.KernelsHyperparams.get_params(
        chosen_kernel, input_dim=n_features, stretch_factor=STRETCH_FACTOR)



    print("##################################################################")
    print(f"**Best kernel selected: {chosen_kernel} with R² = {best_r2:.6f} and MSE = {best_mse:.6f}")
    print("** Kernel Switch - from", INITIAL_KERNEL, "TO:", chosen_kernel)
    print("##################################################################")
    return chosen_kernel, best_kernel_params, best_opt_dict, best_kernel_func, best_kernel_args, best_r2, best_mse, best_r2_tr, best_mse_tr



