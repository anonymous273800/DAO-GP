import numpy as np

# Create a pool of kernels:
# kernels_pool = {
#     "rbf": {"visited": False, "performance": None},
#     "periodic_plus_ard": {"visited": False, "performance": None},
#     "polynomial_degree3": {"visited": False, "performance": None},
#     "log_linear": {"visited": False, "performance": None},
#     "change_point": {"visited": False, "performance": None},
#     "parabolic_plus_linear": {"visited": False, "performance": None},
#     "parabolic_plus_ard": {"visited": False, "performance": None},
#     "periodic_plus_linear": {"visited": False, "performance": None},
#     "composite_parabolic_periodic_ard_rbf": {"visited": False, "performance": None},
#     "composite_periodic_plus_linear": {"visited": False, "performance": None},
# }

kernels_pool = {
    "rbf": {"visited": False, "performance": None},
    "periodic": {"visited": False, "performance": None},
    "rq": {"visited": False, "performance": None},
    "periodic_plus_ard": {"visited": False, "performance": None},
    "polynomial_degree3": {"visited": False, "performance": None},
    "log_linear": {"visited": False, "performance": None},
    "change_point": {"visited": False, "performance": None},
    "parabolic_plus_linear": {"visited": False, "performance": None},
    "composite_parabolic_periodic_ard_rbf": {"visited": False, "performance": None},
    "composite_periodic_plus_linear": {"visited": False, "performance": None}
}

kernels_list = ["rbf", "periodic", "rq", "periodic_plus_ard", "polynomial_degree3", "log_linear", "change_point",
                "parabolic_plus_linear", "composite_parabolic_periodic_ard_rbf", "composite_periodic_plus_linear"]


# New helper: choose a new kernel from the pool.
def choose_new_kernel(kernels_pool):
    # First, try to pick one that hasn't been visited.
    for k, v in kernels_pool.items():
        if not v["visited"]:
            return k
    # If all have been visited, choose the one with best validation R2.
    best_kernel = max(kernels_pool.items(), key=lambda item: item[1]["performance"]["r2"] if item[1]["performance"] is not None else -np.inf)[0]
    return best_kernel