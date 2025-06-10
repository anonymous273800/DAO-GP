import numpy as np
from Utils import Util

# def select_inducing_points(X_win, y_win, K_inv_undecayed, noise, decay_gamma, kernel_func, kernel_args, max_inducing, X_base_time):
#
#
#     if max_inducing is None or len(X_win) <= max_inducing:
#         return X_win, y_win, K_inv_undecayed, X_base_time
#
#     # K_inv_decayed = Util.apply_decay_with_time(K_inv_undecayed, decay_gamma, X_base_time)
#     K_undecayed = kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(X_win))
#     K_decayed = Util.apply_decay_with_time(K_undecayed, decay_gamma, X_base_time)
#     K_inv_decayed = np.linalg.inv(K_decayed)
#     _, stds = Util.computeGP(X_window=X_win,y_window=y_win,K_inv=K_inv_decayed,X_star=X_win,noise=noise, kernel_func=kernel_func, **kernel_args)
#
#     top_idx = np.argsort(-stds)[:max_inducing]
#     X_win_new = X_win[top_idx]
#     y_win_new = y_win[top_idx]
#     X_base_time_new = np.array(X_base_time)[top_idx]
#
#
#     K_win = kernel_func(X_win_new, X_win_new, **kernel_args) + noise * np.eye(len(y_win_new))
#     K_inv_new = np.linalg.inv(K_win)
#
#     return X_win_new, y_win_new, K_inv_new, X_base_time_new

def select_inducing_points(X_win, y_win, X_star, noise, decay_gamma, kernel_func, kernel_args, max_inducing, X_base_time):
    if max_inducing is None or len(X_win) <= max_inducing:
        return X_win, y_win, np.linalg.inv(kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(X_win))), X_base_time

    K_undecayed = kernel_func(X_win, X_win, **kernel_args) + noise * np.eye(len(X_win))
    K_decayed, decay_weights = Util.apply_decay_with_time(K_undecayed, decay_gamma, X_base_time)
    K_inv_decayed = np.linalg.inv(K_decayed)

    # Use separate test points to better observe decay effect

    _, stds = Util.computeGP(X_win, y_win, K_inv_decayed, X_star, noise, kernel_func, **kernel_args)

    # Compute selection scores (e.g., decay-aware uncertainty)
    selection_scores = stds * decay_weights
    top_idx = np.argsort(-selection_scores)[:max_inducing]

    X_win_new = X_win[top_idx]
    y_win_new = y_win[top_idx]
    X_base_time_new = np.array(X_base_time)[top_idx]
    K_new = kernel_func(X_win_new, X_win_new, **kernel_args) + noise * np.eye(len(y_win_new))
    K_inv_new = np.linalg.inv(K_new)

    return X_win_new, y_win_new, K_inv_new, X_base_time_new #, stds, decay_weights, selection_scores, top_idx