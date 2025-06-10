import numpy as np

# Assuming Plotter is available in your environment, otherwise comment out or mock it.
# from Plotter import Plotter

def DS007_ParabolicWave_Simple(
    n_samples: int,
    n_features: int,
    noise_level: float,
    lower_bound: float,
    upper_bound: float,
    stretch_factor: float = 5.0,
    # This parameter controls the influence of additional features, simplified.
    # It applies uniformly to all non-driver features.
    additional_features_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset where the first feature follows a parabolic sine wave,
    and additional features (if present) contribute linearly.

    Parameters:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        noise_level (float): Level of Gaussian noise to add to the target variable.
        lower_bound (float): Minimum value for all features in X.
        upper_bound (float): Maximum value for all features in X.
        stretch_factor (float): Controls the frequency/periodicity of the sine wave
                                 applied to the first feature.
        additional_features_weight (float): Controls how much additional features
                                            (beyond the first) influence the target variable.
                                            Applied linearly to the sum of other features.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target values of shape (n_samples,).
    """
    if n_features <= 0:
        raise ValueError("n_features must be greater than 0.")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be less than upper_bound.")
    if noise_level < 0:
        raise ValueError("noise_level cannot be negative.")

    # 1. Generate X features
    # All features generated independently and uniformly within the same bounds
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))

    # 2. Calculate the core non-linear relationship based on the first feature
    # This is the parabolic sine wave component
    y = X[:, 0] ** 2 * np.sin(stretch_factor * X[:, 0])

    # 3. Add contributions from additional features (if n_features > 1)
    # These features contribute linearly and are summed up.
    if n_features > 1:
        # Sum of all features from the second feature onwards (index 1 to end)
        y += additional_features_weight * X[:, 1:].sum(axis=1)

    # 4. Add Gaussian noise
    y += noise_level * np.random.randn(n_samples)

    return X, y

if __name__ == "__main__":
    print("--- Simple Parabolic Wave Dataset Example ---")

    # Example 1: Single feature, focusing on the core parabolic wave
    X_single, y_single = DS007_ParabolicWave_Simple(
        n_samples=200, n_features=1, noise_level=0.2,
        lower_bound=-5, upper_bound=5, stretch_factor=3
    )
    print(f"Single feature dataset - X shape: {X_single.shape}, y shape: {y_single.shape}")
    print(f"First 5 samples of X_single:\n{X_single[:5].flatten()}")
    print(f"First 5 samples of y_single:\n{y_single[:5]}")
    # try:
    #     Plotter.plot_basic(X_single, y_single, title="Single Feature Parabolic Wave")
    # except NameError:
    #     print("Plotter not available. Skipping plot.")


    # Example 2: Multi-feature dataset with additional features contributing linearly
    X_multi, y_multi = DS007_ParabolicWave_Simple(
        n_samples=500, n_features=3, noise_level=0.1,
        lower_bound=-10, upper_bound=10, stretch_factor=4,
        additional_features_weight=0.3
    )
    print(f"\nMulti-feature dataset - X shape: {X_multi.shape}, y shape: {y_multi.shape}")
    print(f"First 5 samples of X_multi:\n{X_multi[:5]}")
    print(f"First 5 samples of y_multi:\n{y_multi[:5]}")
    # try:
    #     # If you have a 3D plotter:
    #     # Plotter.plot_3d(X_multi[:, 0], X_multi[:, 1], y_multi, title="Multi-Feature Dataset (X0 vs Y)")
    #     # Or just plot the driver feature:
    #     Plotter.plot_basic(X_multi[:, 0], y_multi, title="Multi-Feature Dataset (X0 vs Y)")
    # except NameError:
    #     print("Plotter not available. Skipping plot.")

    # Example 3: Different noise level
    X_noisy, y_noisy = DS007_ParabolicWave_Simple(
        n_samples=100, n_features=2, noise_level=0.5,
        lower_bound=-7, upper_bound=7, stretch_factor=2
    )
    print(f"\nNoisy dataset - X shape: {X_noisy.shape}, y shape: {y_noisy.shape}")
    # try:
    #     Plotter.plot_basic(X_noisy[:, 0], y_noisy, title="Noisy Parabolic Wave (X0 vs Y)")
    # except NameError:
    #     print("Plotter not available. Skipping plot.")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.scatter(X_single, y_single, label='Data Points',c=y_single.flatten(), cmap="coolwarm", alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()