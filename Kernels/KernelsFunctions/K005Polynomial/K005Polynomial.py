import numpy as np

# ==== New Degree-3 Polynomial Kernel ====
def polynomial_kernel(X1, X2, coef0, scaling_factor):
    # degree = int(degree)
    # print("*****************************", degree, "***************************")
    """
    Computes a degree-3 polynomial kernel:
       k(x,x') = (scaling_factor * <x,x'> + coef0)^3
    Parameters:
      X1 : (n x d) array
      X2 : (m x d) array
      coef0 : constant term
      scaling_factor : scaling factor
    Returns:
      (n x m) kernel matrix.
    """
    return (scaling_factor * np.dot(X1, X2.T) + coef0) ** 3
