import numpy as np
import matplotlib.pyplot as plt
from Utils import Util
from Utils import Constants
import seaborn as sns
from matplotlib.lines import Line2D

def plot_basic(X, y):
    n_samples, n_features = X.shape
    if n_features == 1:
        plt.figure(figsize=(10, 7))
        plt.scatter(X, y, label='Data Points',c=y.flatten(), cmap="coolwarm", alpha=0.8)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    if n_features == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y.flatten(), c=y.flatten(), cmap="coolwarm", alpha=0.8)

        ax.set_xlabel("Feature 1 (X1)")
        ax.set_ylabel("Feature 2 (X2)")
        ax.set_zlabel("Target (y)")

        plt.show()


def plot_abrupt_drift_old(X1, y1, X2, y2):
    n_samples, n_features = X1.shape
    if n_features == 1:
        plt.figure(figsize=(10, 7))


        # Updated color scheme: Soft green and plum
        plt.scatter(X1, y1, color="#8da0cb", label="Before Drift - Concept 1", alpha=0.4,
                    linewidths=0.3)
        plt.scatter(X2, y2, color="#66c2a5", label="After Drift - Concept 2", alpha=0.4,
                    linewidths=0.3)

        plt.xlabel('X')
        plt.ylabel('y')
        # Legend: change location here
        plt.legend(
            loc='upper left',
            fontsize=6,  # Font size
            markerscale=0.8,  # Shrink legend markers
            labelspacing=0.2,  # Vertical space between labels
            handletextpad=0.4,  # Padding between handle and text
            borderaxespad=0.2,  # Padding between legend and plot
            borderpad=0.3,  # Padding inside the legend border
        )
        plt.show()

    if n_features == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X1[:, 0], X1[:, 1], y1.flatten(), c=y1.flatten(), cmap="coolwarm", alpha=0.8)
        ax.scatter(X2[:, 0], X2[:, 1], y2.flatten(), c=y2.flatten(), cmap="coolwarm", alpha=0.8)
        ax.set_xlabel("Feature 1 (X1)")
        ax.set_ylabel("Feature 2 (X2)")
        ax.set_zlabel("Target (y)")
        plt.show()


import seaborn as sns # Import seaborn for better aesthetics

# def plot_abrupt_drift(X1, y1, X2, y2):
#     """
#     Generates a professional-looking plot to visualize abrupt concept drift.
#
#     Parameters:
#     X1 (np.array): Feature data before drift (Concept 1).
#     y1 (np.array): Target data before drift (Concept 1).
#     X2 (np.array): Feature data after drift (Concept 2).
#     y2 (np.array): Target data after drift (Concept 2).
#     """
#     n_samples, n_features = X1.shape
#
#     # Apply a professional seaborn style for better aesthetics
#     sns.set_style("whitegrid") # Or "darkgrid", "white", "dark", "ticks"
#     # sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5}) # Adjust context for overall scaling
#
#     if n_features == 1:
#         plt.figure(figsize=(10, 6)) # Slightly adjusted figure size for common aspect ratios
#         ax = plt.gca() # Get current axes for more control
#
#         # Professional color palette (e.g., from viridis, plasma, cividis, or custom)
#         # Using a distinct, modern palette from Matplotlib colormaps or custom hex codes
#         color_concept1 = "#4c72b0"  # A nice blue
#         color_concept2 = "#66c2a5"#"#dd8452"  # A nice orange/red
#
#         # Improved scatter plot aesthetics
#         plt.scatter(X1, y1, color=color_concept1, label="Before Drift (Concept 1)",
#                     alpha=0.6, s=30, edgecolors='w', linewidths=0.5, zorder=2) # s=marker size, edgecolors for crispness
#         plt.scatter(X2, y2, color=color_concept2, label="After Drift (Concept 2)",
#                     alpha=0.6, s=30, edgecolors='w', linewidths=0.5, zorder=2)
#
#         # Enhance axes labels and title
#         plt.xlabel('X', fontsize=12)
#         plt.ylabel('y', fontsize=12)
#         plt.title('Abrupt Concept Drift Visualization (1D)', fontsize=14, fontweight='bold')
#
#         # Customize ticks and grid
#         ax.tick_params(axis='both', which='major', labelsize=10)
#         ax.grid(True, linestyle='--', alpha=0.6, zorder=0) # Zorder places grid behind points
#
#         # Legend customization (more refined)
#         plt.legend(
#             loc='upper left',
#             fontsize=10, # Slightly larger font for readability
#             markerscale=1.0, # Adjust marker scale
#             frameon=True, # Add a frame around the legend
#             edgecolor='black', # Legend frame color
#             fancybox=True, # Rounded corners for the legend box
#             shadow=False, # No shadow for cleaner look
#             borderpad=0.5, # Padding inside the legend border
#             labelspacing=0.8, # More space between labels for clarity
#         )
#
#         # Remove top and right spines for a cleaner look
#         sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
#
#         plt.tight_layout() # Adjust layout to prevent labels from overlapping
#         plt.show()
#
#     if n_features == 2:
#         fig = plt.figure(figsize=(12, 9)) # Larger figure for 3D plot
#         ax = fig.add_subplot(111, projection='3d')
#
#         # Choose a perceptually uniform colormap (e.g., 'viridis', 'plasma', 'cividis')
#         # 'coolwarm' is good for diverging data. For general target values, 'viridis' or 'plasma' are often preferred.
#         cmap_concept1 = 'viridis'
#         cmap_concept2 = 'plasma'
#
#         # Scatter plot for 3D data with color mapping
#         # `s` for marker size, `alpha` for transparency
#         scatter1 = ax.scatter(X1[:, 0], X1[:, 1], y1.flatten(),
#                               c=y1.flatten(), cmap=cmap_concept1, alpha=0.7, s=40, label="Before Drift")
#         scatter2 = ax.scatter(X2[:, 0], X2[:, 1], y2.flatten(),
#                               c=y2.flatten(), cmap=cmap_concept2, alpha=0.7, s=40, label="After Drift")
#
#         # Set labels with clear font styles
#         ax.set_xlabel("($X_1$)", fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_ylabel("($X_2$)", fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_zlabel("y", fontsize=12, fontweight='bold', labelpad=10)
#         ax.set_title('Abrupt Concept Drift Visualization (2D)', fontsize=14, fontweight='bold')
#
#         # Add colorbars for clarity (important for 3D scatter plots with color mapping)
#         fig.colorbar(scatter1, ax=ax, pad=0.1, shrink=0.5, aspect=10, label="y")
#         fig.colorbar(scatter2, ax=ax, pad=0.15, shrink=0.5, aspect=10, label="y")
#
#
#         # Customize the view angle for better visualization (optional, play with these values)
#         ax.view_init(elev=20, azim=-60) # elev=elevation angle, azim=azimuth angle
#
#         # Improve background and grid for 3D
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False
#         ax.grid(True, linestyle='--', alpha=0.5)
#
#         # Add a legend for the scatter groups
#         ax.legend(
#             loc='upper right',
#             fontsize=10,
#             markerscale=1.0,
#             frameon=True,
#             edgecolor='black',
#             fancybox=True,
#             shadow=False,
#             borderpad=0.5,
#             labelspacing=0.8,
#         )
#
#         plt.tight_layout()
#         plt.show()





def plot_abrupt_drift(X1, y1, X2, y2, dataset_type):
    """
    Generates a professional-looking plot to visualize abrupt concept drift,
    with an added dataset label at the top of the legend.

    Parameters:
    X1 (np.array): Feature data before drift (Concept 1).
    y1 (np.array): Target data before drift (Concept 1).
    X2 (np.array): Feature data after drift (Concept 2).
    y2 (np.array): Target data after drift (Concept 2).
    """
    n_samples, n_features = X1.shape

    sns.set_style("whitegrid")

    if n_features == 1:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        color_concept1 = "#4c72b0"  # A nice blue
        color_concept2 = "#66c2a5"  # A nice green

        # --- LINES ADDED/MODIFIED TO CONTROL LEGEND ORDER (START) ---
        legend_handles = []
        legend_labels = []

        # 1. Add the "Dataset" label first
        dataset_handle, = ax.plot([], [], ' ', label=rf'Dataset: {dataset_type}')
        legend_handles.append(dataset_handle)
        legend_labels.append(rf'Dataset: {dataset_type}')
        # --- LINES ADDED/MODIFIED TO CONTROL LEGEND ORDER (END) ---

        # Improved scatter plot aesthetics
        # --- LINES MODIFIED TO CAPTURE SCATTER HANDLES ---
        scatter1_handle = plt.scatter(X1, y1, color=color_concept1, label="Before Drift (Concept 1)",
                                      alpha=0.6, s=30, edgecolors='w', linewidths=0.5, zorder=2)
        scatter2_handle = plt.scatter(X2, y2, color=color_concept2, label="After Drift (Concept 2)",
                                      alpha=0.6, s=30, edgecolors='w', linewidths=0.5, zorder=2)

        # --- LINES ADDED TO CONTROL LEGEND ORDER (START) ---
        legend_handles.append(scatter1_handle)
        legend_labels.append("Before Drift (Concept 1)")
        legend_handles.append(scatter2_handle)
        legend_labels.append("After Drift (Concept 2)")
        # --- LINES ADDED TO CONTROL LEGEND ORDER (END) ---


        # Enhance axes labels and title
        plt.xlabel('X', fontsize=12, fontweight='bold') # Added fontweight
        plt.ylabel('y', fontsize=12, fontweight='bold') # Added fontweight
        plt.title('Abrupt Concept Drift Visualization (1D)', fontsize=14, fontweight='bold')

        # Customize ticks and grid
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

        # Legend customization (more refined) - NOW USES EXPLICIT HANDLES AND LABELS
        plt.legend(
            handles=legend_handles, # Pass the collected handles
            labels=legend_labels,   # Pass the collected labels
            loc='upper left',
            fontsize=10,
            markerscale=1.0,
            frameon=True,
            edgecolor='black',
            fancybox=True,
            shadow=False,
            borderpad=0.5,
            labelspacing=0.8
            # ,
            # title="Drift Information" # Optional: Add a title to the legend
        )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        plt.tight_layout()
        plt.show()

    if n_features == 2:
        fig = plt.figure(figsize=(12, 9)) # Larger figure for 3D plot
        ax = fig.add_subplot(111, projection='3d')

        # --- LINES ADDED/MODIFIED FOR 3D LEGEND ORDER (START) ---
        legend_handles = []
        legend_labels = []

        dataset_type_3d = 'Abrupt Drift' # You can customize this string
        dataset_handle_3d, = ax.plot([], [], [], ' ', label=rf'Dataset: {dataset_type_3d}') # For 3D, need 3 empty lists
        legend_handles.append(dataset_handle_3d)
        legend_labels.append(rf'Dataset: {dataset_type_3d}')
        # --- LINES ADDED/MODIFIED FOR 3D LEGEND ORDER (END) ---

        # Using distinct colormaps for different concepts might be better than
        # `coolwarm` which is diverging. For comparing two sets, distinct hues are good.
        cmap_concept1 = 'Blues_r' # Reverse blues
        cmap_concept2 = 'Oranges_r' # Reverse oranges

        # Scatter plot for 3D data with color mapping
        # --- LINES MODIFIED TO CAPTURE SCATTER HANDLES ---
        scatter1 = ax.scatter(X1[:, 0], X1[:, 1], y1.flatten(),
                              c=y1.flatten(), cmap=cmap_concept1, alpha=0.7, s=40, label="Before Drift")
        scatter2 = ax.scatter(X2[:, 0], X2[:, 1], y2.flatten(),
                              c=y2.flatten(), cmap=cmap_concept2, alpha=0.7, s=40, label="After Drift")

        # --- LINES ADDED TO CONTROL LEGEND ORDER (START) ---
        legend_handles.append(scatter1)
        legend_labels.append("Before Drift (Concept 1)")
        legend_handles.append(scatter2)
        legend_labels.append("After Drift (Concept 2)")
        # --- LINES ADDED TO CONTROL LEGEND ORDER (END) ---

        # Set labels with clear font styles
        ax.set_xlabel("Feature 1 ($X_1$)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Feature 2 ($X_2$)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel("Target (y)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Abrupt Concept Drift Visualization (2D)', fontsize=14, fontweight='bold')

        # Add colorbars for clarity (important for 3D scatter plots with color mapping)
        # Note: If you want these colorbars in the legend too, it's more complex,
        # typically colorbars are separate.
        fig.colorbar(scatter1, ax=ax, pad=0.1, shrink=0.5, aspect=10, label="Target (y) - Before Drift")
        fig.colorbar(scatter2, ax=ax, pad=0.15, shrink=0.5, aspect=10, label="Target (y) - After Drift")


        # Customize the view angle for better visualization (optional)
        ax.view_init(elev=20, azim=-60)

        # Improve background and grid for 3D
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add a legend for the scatter groups
        ax.legend(
            handles=legend_handles, # Pass the collected handles
            labels=legend_labels,   # Pass the collected labels
            loc='upper right',
            fontsize=10,
            markerscale=1.0,
            frameon=True,
            edgecolor='black',
            fancybox=True,
            shadow=False,
            borderpad=0.5,
            labelspacing=0.8,
            title="Drift Information" # Optional: Add a title to the legend
        )

        plt.tight_layout()
        plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_incremental_drift_old(*datasets):
    """
    Plots multiple datasets in 2D or 3D based on the number of features.

    Parameters:
        *datasets: Each dataset is a tuple (X, y).
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]  # Distinct colors

    # Determine dimensionality based on the first dataset
    n_features = datasets[0][0].shape[1]

    if n_features == 1:
        plt.figure(figsize=(10, 7))

        # Iterate over all (X, y) pairs and plot with different colors
        for i, (X, y) in enumerate(datasets):
            plt.scatter(X, y, color=colors[i % len(colors)], label=f"Concept {i + 1}", alpha=0.6)

        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    elif n_features == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i, (X, y) in enumerate(datasets):
            ax.scatter(X[:, 0], X[:, 1], y.flatten(), color=colors[i % len(colors)], label=f"Concept {i + 1}",
                       alpha=0.8)

        ax.set_xlabel("Feature 1 (X1)")
        ax.set_ylabel("Feature 2 (X2)")
        ax.set_zlabel("Target (y)")
        ax.legend()
        plt.show()





def plot_incremental_drift(*datasets, dataset_type):

    if len(datasets) <= 10:
        plot_colors = sns.color_palette("tab10", n_colors=len(datasets))
    elif len(datasets) <= 20:
        plot_colors = sns.color_palette("tab20", n_colors=len(datasets))
    else:
        plot_colors = sns.color_palette("tab20", n_colors=len(datasets) % 20 + 1)

    sns.set_style("whitegrid")

    if not datasets:
        print("No datasets provided for plotting.")
        return

    n_features = datasets[0][0].shape[1]

    if n_features == 1:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Store handles and labels to control legend order
        legend_handles = []
        legend_labels = []

        # 1. Add the "Dataset" label first
        # Plot an invisible item to get a handle for the legend
        dataset_handle, = ax.plot([], [], ' ', label=rf'Dataset: {dataset_type}') # Note the comma for unpacking
        legend_handles.append(dataset_handle)
        legend_labels.append(rf'Dataset: {dataset_type}')


        # 2. Iterate over all (X, y) pairs and plot with different colors
        for i, (X, y) in enumerate(datasets):
            scatter_handle = ax.scatter(X, y, color=plot_colors[i % len(plot_colors)],
                                        label=f"Concept {i + 1}",
                                        alpha=0.6, s=30, edgecolors='w', linewidths=0.5, zorder=2)
            legend_handles.append(scatter_handle)
            legend_labels.append(f"Concept {i + 1}")


        # Enhanced labels and title
        plt.xlabel('X', fontsize=12, fontweight='bold')
        plt.ylabel('y', fontsize=12, fontweight='bold')
        plt.title('Incremental Concept Drift Visualization (1D)', fontsize=14, fontweight='bold')

        # Customize ticks and grid
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

        # Explicitly pass handles and labels to control legend order
        plt.legend(handles=legend_handles, labels=legend_labels)

        # Remove top and right spines for a cleaner look
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        plt.tight_layout()
        plt.show()

    elif n_features == 2:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # For 3D, if you want a "Dataset" label at the top, you'd apply a similar method
        # (create empty plot for handle/label, then add scatter handles/labels)
        legend_handles = []
        legend_labels = []

        # Add the "Dataset" label for 3D if desired
        dataset_type_3d = 'Inc. Drift Sin' # Or adjust for 3D context
        dataset_handle_3d, = ax.plot([], [], [], ' ', label=rf'Dataset: {dataset_type_3d}')
        legend_handles.append(dataset_handle_3d)
        dataset_3d_type = "dataset 3d type"
        legend_labels.append(rf'Dataset: {dataset_3d_type}')


        for i, (X, y) in enumerate(datasets):
            scatter_handle = ax.scatter(X[:, 0], X[:, 1], y.flatten(),
                                       color=plot_colors[i % len(plot_colors)],
                                       label=f"Concept {i + 1}",
                                       alpha=0.7, s=40, edgecolors='w', linewidths=0.2)
            legend_handles.append(scatter_handle)
            legend_labels.append(f"Concept {i + 1}")


        ax.set_xlabel("Feature 1 (X1)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Feature 2 (X2)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel("Target (y)", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Incremental Concept Drift Visualization (2D)', fontsize=14, fontweight='bold')

        ax.view_init(elev=20, azim=-60)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linestyle='--', alpha=0.5)

        # Explicitly pass handles and labels to control legend order for 3D
        ax.legend(handles=legend_handles, labels=legend_labels)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Unsupported number of features: {n_features}. Only 1D and 2D supported for plotting.")


# def plot(X_base, y_base, K_inv_undecayed, gamma, X_test, y_test, noise, length_scale, sigma_f):
def plot(X,y, mu, std, title, label, color='blue', alpha=.4):
    sorted_idx = np.argsort(X[:, 0])
    X = X[sorted_idx]
    y = y[sorted_idx]

    # mu_test, std_test = predict_sliding_gp(X_base, y_base, K_inv_undecayed,
    #                                        X_test, noise, length_scale, sigma_f, gamma)
    mu = mu[sorted_idx]
    std = std[sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, color=color, alpha=alpha, label=label)
    plt.plot(X, mu, 'r-', label='Online GP Mean')
    plt.fill_between(X.ravel(),
                     mu - 2 * std,
                     mu + 2 * std,
                     color='r', alpha=0.2, label='±2 std')
    plt.title(title)
    plt.legend()
    plt.show()




from Utils import Util

import matplotlib.pyplot as plt
import numpy as np
from Utils import Util

def plot_final_old(X_win, y_win, K_inv_undecayed, gamma, X_test, y_test, noise, kernel_func,final_mse, final_r2, **kernel_args):
    import matplotlib.pyplot as plt

    if X_test.shape[1] != 1:
        print("Skipping plot: only works for 1D input.")
        return

    sorted_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_idx]
    y_test_sorted = y_test[sorted_idx]

    mu_test, std_test = Util.computeGP(
        X_window=X_win,
        y_window=y_win,
        K_inv=K_inv_undecayed,
        X_star=X_test,
        noise=noise,
        kernel_func=kernel_func,
        **kernel_args
    )

    mu_test_sorted = mu_test[sorted_idx]
    std_test_sorted = std_test[sorted_idx]
    # std_test_sorted = std_test_sorted + 1 # add .1 if std is too small

    plt.figure(figsize=(8, 4))

    # Test data
    plt.scatter(X_test_sorted, y_test_sorted,
                color='blue', linewidth=0.4,
                alpha=0.5, s=10, label='Test Data', marker='o', zorder=2)

    # Inducing points
    plt.scatter(X_win, y_win,
                color='green', alpha=0.5, s=10,
                label='Inducing Points', linewidth=0.4, zorder=3)




    # GP mean
    plt.plot(X_test_sorted, mu_test_sorted,
             color='forestgreen', label='Online GP Mean', linewidth=2.0, zorder=4)

    # Confidence region
    plt.fill_between(X_test_sorted.ravel(),
                     mu_test_sorted - 2 * std_test_sorted,
                     mu_test_sorted + 2 * std_test_sorted,
                     color='thistle', alpha=0.5, label='±2 std', zorder=1)

    # Labels and title
    plt.title("Final Online GP Prediction")
    plt.xlabel("Input Feature")
    plt.ylabel("Target Value")

    # Add results to the legend
    plt.plot([], [], ' ', label=f'$\gamma$ = {gamma}')
    plt.plot([], [], ' ', label=f'MSE = {final_mse:.4f}')
    plt.plot([], [], ' ', label=f'R² = {final_r2:.4f}')


    # Legend: change location here
    plt.legend(
        loc='upper left',
        fontsize=6,  # Font size
        markerscale=0.8,  # Shrink legend markers
        labelspacing=0.2,  # Vertical space between labels
        handletextpad=0.4,  # Padding between handle and text
        borderaxespad=0.2,  # Padding between legend and plot
        borderpad=0.3,  # Padding inside the legend border
    )

    # Optional: Grid and layout
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # # Test data
    # plt.scatter(X_test_sorted, y_test_sorted, color='blue', linewidth=0.5, edgecolor='black', alpha=0.6, s=40, label='Test Data', marker='+')
    #
    # # Inducing points
    # plt.scatter(X_win, y_win, color='green', alpha=0.6, s=40, label='Inducing Points', edgecolor='black', linewidth=0.5)
    #
    # plt.plot(X_test_sorted, mu_test_sorted, color='forestgreen', label='Online GP Mean')
    # plt.fill_between(X_test_sorted.ravel(),
    #                  mu_test_sorted - 2 * std_test_sorted,
    #                  mu_test_sorted + 2 * std_test_sorted,
    #                  color='thistle', alpha=0.3, label='±2 std')
    # plt.title("Final Online GP Prediction")
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'Util' is a module you have for computeGP.
# Make sure it's correctly imported or available in your environment.

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'Util' is a module you have for computeGP.
# Make sure it's correctly imported or available in your environment.

def plot_final(X_win, y_win, K_inv_undecayed, gamma, dataset_type ,X_test, y_test, noise, kernel_func, final_mse, final_r2, **kernel_args):
    """
    Generates a professional-quality 2D plot for visualizing Online Gaussian Process
    regression results, including predictions, uncertainty, and inducing points.

    Args:
        X_win (np.ndarray): Array of inducing points (features).
        y_win (np.ndarray): Array of target values corresponding to inducing points.
        K_inv_undecayed (np.ndarray): Inverse of the kernel matrix for inducing points.
        gamma (float): Decay factor (gamma) value, displayed on the plot.
        X_test (np.ndarray): Array of test data features for prediction.
        y_test (np.ndarray): Array of true target values for the test data.
        noise (float): Noise variance parameter for the GP.
        kernel_func (callable): The kernel function used by the GP.
        final_mse (float): Mean Squared Error of the model's predictions.
        final_r2 (float): R-squared value of the model's predictions.
        **kernel_args: Additional keyword arguments passed to the kernel function.
    """

    if X_test.shape[1] != 1:
        print(f"Warning: Skipping plot for input dimensionality {X_test.shape[1]}. "
              "Plotting function only supports 1D input features.")
        return

    sorted_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_idx]
    y_test_sorted = y_test[sorted_idx]

    try:
        mu_test, std_test = Util.computeGP(
            X_window=X_win,
            y_window=y_win,
            K_inv=K_inv_undecayed,
            X_star=X_test_sorted,
            noise=noise,
            kernel_func=kernel_func,
            **kernel_args
        )

        mu_test_sorted = mu_test
        std_test_sorted = std_test
    except NameError:
        print("Error: 'Util.computeGP' function not found. Please ensure 'Util' module is correctly imported and 'computeGP' is defined.")
        return
    except Exception as e:
        print(f"Error computing GP predictions: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([], [], ' ', label=rf'Dataset: {dataset_type}', zorder=1)
    ax.fill_between(
        X_test_sorted.ravel(),
        mu_test_sorted - 2 * std_test_sorted,
        mu_test_sorted + 2 * std_test_sorted,
        color='thistle',
        alpha=0.4,
        label='$\pm 2$ std',
        zorder=2
    )

    ax.plot(
        X_test_sorted,
        mu_test_sorted,
        color='forestgreen',
        label='Online GP Mean',
        linewidth=2.5,
        zorder=3
    )

    ax.scatter(
        X_test_sorted,
        y_test_sorted,
        color='royalblue',
        alpha=0.5,
        s=13,
        label='Test Data',
        marker='o',
        edgecolors='none',
        zorder=4
    )

    ax.scatter(
        X_win,
        y_win,
        color='darkorange',
        alpha=0.7,
        s=25,
        label='Inducing Points',
        marker='^',
        edgecolors='black',
        linewidth=0.5,
        zorder=5
    )

    ax.set_title("Final Online GP Prediction")
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Target Value")

    # --- CHANGED LINES TO MAKE TEXT BOLDER ---
    # ax.plot([], [], ' ', label=r'$\mathbf{\gamma} = \mathbf{' + f'{gamma}' + r'}$')
    ax.plot([], [], ' ', label=r'$\mathbf{MSE} = \mathbf{' + f'{final_mse:.4f}' + r'}$')
    ax.plot([], [], ' ', label=r'$\mathbf{R^2} = \mathbf{' + f'{final_r2:.4f}' + r'}$')

    ax.legend(
        loc='upper left',
        fontsize=8,
        markerscale=0.8,
        labelspacing=0.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        borderpad=0.3,
        frameon=True,
        edgecolor='gray',
        fancybox=True,
        shadow=False
    )

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def plot_final_show_decay(X_win, y_win, K_inv_undecayed, gamma, dataset_type ,X_test, y_test, noise, kernel_func, final_mse, final_r2,
                          xticks_lower=-1.5, xticks_higher=1.5, yticks_lower=1.5, yticks_hight=1.5, **kernel_args):
    """
    Generates a professional-quality 2D plot for visualizing Online Gaussian Process
    regression results, including predictions, uncertainty, and inducing points.

    Args:
        X_win (np.ndarray): Array of inducing points (features).
        y_win (np.ndarray): Array of target values corresponding to inducing points.
        K_inv_undecayed (np.ndarray): Inverse of the kernel matrix for inducing points.
        gamma (float): Decay factor (gamma) value, displayed on the plot.
        X_test (np.ndarray): Array of test data features for prediction.
        y_test (np.ndarray): Array of true target values for the test data.
        noise (float): Noise variance parameter for the GP.
        kernel_func (callable): The kernel function used by the GP.
        final_mse (float): Mean Squared Error of the model's predictions.
        final_r2 (float): R-squared value of the model's predictions.
        **kernel_args: Additional keyword arguments passed to the kernel function.
    """

    if X_test.shape[1] != 1:
        print(f"Warning: Skipping plot for input dimensionality {X_test.shape[1]}. "
              "Plotting function only supports 1D input features.")
        return

    sorted_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_idx]
    y_test_sorted = y_test[sorted_idx]

    try:
        mu_test, std_test = Util.computeGP(
            X_window=X_win,
            y_window=y_win,
            K_inv=K_inv_undecayed,
            X_star=X_test_sorted,
            noise=noise,
            kernel_func=kernel_func,
            **kernel_args
        )

        mu_test_sorted = mu_test
        std_test_sorted = std_test
    except NameError:
        print("Error: 'Util.computeGP' function not found. Please ensure 'Util' module is correctly imported and 'computeGP' is defined.")
        return
    except Exception as e:
        print(f"Error computing GP predictions: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot([], [], ' ', label=rf'Dataset: {dataset_type}', zorder=1)
    ax.fill_between(
        X_test_sorted.ravel(),
        mu_test_sorted - 2 * std_test_sorted,
        mu_test_sorted + 2 * std_test_sorted,
        color='thistle',
        alpha=0.4,
        label='$\pm 2$ std',
        zorder=2
    )

    ax.plot(
        X_test_sorted,
        mu_test_sorted,
        color='forestgreen',
        label='Online GP Mean',
        linewidth=2.5,
        zorder=3
    )

    ax.scatter(
        X_test_sorted,
        y_test_sorted,
        color='royalblue',
        alpha=0.5,
        s=13,
        label='Test Data',
        marker='o',
        edgecolors='none',
        zorder=4
    )

    ax.scatter(
        X_win,
        y_win,
        color='darkorange',
        alpha=0.7,
        s=25,
        label='Inducing Points',
        marker='^',
        edgecolors='black',
        linewidth=0.5,
        zorder=5
    )

    ax.set_title("Final Online GP Prediction")
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Target Value")

    # --- CHANGED LINES TO MAKE TEXT BOLDER ---
    # ax.plot([], [], ' ', label=r'$\mathbf{\gamma} = \mathbf{' + f'{gamma}' + r'}$')
    gamma_label = (
        rf'$\mathbf{{\gamma}} = \mathbf{{{gamma}}} \, (\text{{no decay}})$'
        if gamma == 1 else
        rf'$\mathbf{{\gamma}} = \mathbf{{{gamma}}}$'
    )
    ax.plot([], [], ' ', label=gamma_label)

    ax.plot([], [], ' ', label=r'$\mathbf{MSE} = \mathbf{' + f'{final_mse:.4f}' + r'}$')
    ax.plot([], [], ' ', label=r'$\mathbf{R^2} = \mathbf{' + f'{final_r2:.4f}' + r'}$')

    ax.legend(
        loc='upper left',
        fontsize=8,
        markerscale=0.8,
        labelspacing=0.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        borderpad=0.3,
        frameon=True,
        edgecolor='gray',
        fancybox=True,
        shadow=False
    )

    # Concatenate all x and y values to determine global axis limits
    ax.set_xlim(xticks_lower, xticks_higher)
    ax.set_ylim(yticks_lower, yticks_hight)


    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'Util' is a module you have for computeGP.
# Make sure it's correctly imported or available in your environment.

def plot_final_mini(X_win, y_win, K_inv_undecayed, gamma, X_test, y_test, noise, kernel_func, final_mse, final_r2, **kernel_args):
    """
    Generates a professional-quality 2D plot for visualizing Online Gaussian Process
    regression results, including predictions, uncertainty, and inducing points.

    Args:
        X_win (np.ndarray): Array of inducing points (features).
        y_win (np.ndarray): Array of target values corresponding to inducing points.
        K_inv_undecayed (np.ndarray): Inverse of the kernel matrix for inducing points.
        gamma (float): Decay factor (gamma) value, displayed on the plot.
        X_test (np.ndarray): Array of test data features for prediction.
        y_test (np.ndarray): Array of true target values for the test data.
        noise (float): Noise variance parameter for the GP.
        kernel_func (callable): The kernel function used by the GP.
        final_mse (float): Mean Squared Error of the model's predictions.
        final_r2 (float): R-squared value of the model's predictions.
        **kernel_args: Additional keyword arguments passed to the kernel function.
    """

    if X_test.shape[1] != 1:
        print(f"Warning: Skipping plot for input dimensionality {X_test.shape[1]}. "
              "Plotting function only supports 1D input features.")
        return

    sorted_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_idx]
    y_test_sorted = y_test[sorted_idx]

    try:
        mu_test, std_test = Util.computeGP(
            X_window=X_win,
            y_window=y_win,
            K_inv=K_inv_undecayed,
            X_star=X_test_sorted,
            noise=noise,
            kernel_func=kernel_func,
            **kernel_args
        )

        mu_test_sorted = mu_test
        std_test_sorted = std_test
    except NameError:
        print("Error: 'Util.computeGP' function not found. Please ensure 'Util' module is correctly imported and 'computeGP' is defined.")
        return
    except Exception as e:
        print(f"Error computing GP predictions: {e}")
        return

    # --- LINE TO UPDATE ---
    # Target pixels: 420 width x 210 height
    # Choose a DPI. 96 DPI is common for screen/web. For higher quality print, use 300.
    target_dpi = 96
    width_inches = 420 / target_dpi
    height_inches = 210 / target_dpi
    fig, ax = plt.subplots(figsize=(width_inches, height_inches)) # CHANGED LINE

    ax.fill_between(
        X_test_sorted.ravel(),
        mu_test_sorted - 2 * std_test_sorted,
        mu_test_sorted + 2 * std_test_sorted,
        color='thistle',
        alpha=0.4,
        label='$\pm 2$ std',
        zorder=1
    )

    ax.plot(
        X_test_sorted,
        mu_test_sorted,
        color='forestgreen',
        label='Online GP Mean',
        linewidth=2.5,
        zorder=2
    )

    ax.scatter(
        X_test_sorted,
        y_test_sorted,
        color='royalblue',
        alpha=0.8,
        s=20,
        label='Test Data',
        marker='o',
        edgecolors='none',
        zorder=4
    )

    ax.scatter(
        X_win,
        y_win,
        color='darkorange',
        alpha=0.7,
        s=25,
        label='Inducing Points',
        marker='^',
        edgecolors='black',
        linewidth=0.5,
        zorder=3
    )

    ax.set_title("Final Online GP Prediction")
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Target Value")

    ax.plot([], [], ' ', label=r'$\mathbf{\gamma} = \mathbf{' + f'{gamma}' + r'}$')
    ax.plot([], [], ' ', label=r'$\mathbf{MSE} = \mathbf{' + f'{final_mse:.4f}' + r'}$')
    ax.plot([], [], ' ', label=r'$\mathbf{R^2} = \mathbf{' + f'{final_r2:.4f}' + r'}$')

    ax.legend(
        loc='upper left',
        fontsize=8,
        markerscale=0.8,
        labelspacing=0.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        borderpad=0.3,
        frameon=True,
        edgecolor='gray',
        fancybox=True,
        shadow=False
    )

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- OPTIONAL: Add saving logic here if you want to control output pixel size ---
    # Since you didn't want to change the signature, you could hardcode a path
    # or define a global flag.
    # For example, to save it as "my_plot_420x210.png":
    # plt.savefig("my_plot_420x210.png", dpi=target_dpi, bbox_inches='tight')
    # plt.close(fig) # Close the figure if saving and not showing

    plt.show() # Original behavior


def plot_minibatches_results_standard(x_axis_dao_gp, y_axis_dao_gp, kpi, label_dao_gp):
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp, linestyle='-', marker='o', markersize=3, linewidth=2,
                      label=label_dao_gp, zorder=10)

    # Adding labels and title


    plt.xlabel('$N$', fontsize=7)
    if kpi == 'R2': plt.ylabel('R2', fontsize=7)
    if kpi == 'MSE': plt.ylabel('MSE', fontsize=7)

    plt.title('Performance', fontsize=7)

    # Adjust font size of numbers on x-axis and y-axis
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    # Customize grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Only vertical lines with dashed style

    # Remove top and right spines
    sns.despine()

    # Adding grid
    plt.grid(True)

    # Show plot
    plt.show()


def plot_minibatches_results_standard_all(x_axis_dao_gp, y_axis_dao_gp, kpi, label_dao_gp,
                                          x_axis_fitc, y_axis_fitc, label_fitc,
                                          x_axis_kpa, y_axis_kpa, label_kpa,
                                          legend_loc):
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp, linestyle='-', marker='.', markersize=2, linewidth=1,
                      label=label_dao_gp, zorder=10)
    line2, = plt.plot(x_axis_fitc, y_axis_fitc, linestyle='-', marker='.', markersize=2, linewidth=1,
                      label=label_fitc, zorder=9)
    line5, = plt.plot(x_axis_kpa, y_axis_kpa, linestyle='-', marker='.', markersize=2, linewidth=1,
                      label=label_kpa, zorder=6)


    # Adding labels and title


    plt.xlabel('$N$', fontsize=7)
    if kpi == 'R2': plt.ylabel('R2', fontsize=7)
    if kpi == 'MSE': plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    # Adjust font size of numbers on x-axis and y-axis
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    # Customize grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Only vertical lines with dashed style

    # Remove top and right spines
    sns.despine()

    # Adding grid
    plt.grid(True)

    plt.legend(fontsize='small', loc=legend_loc, fancybox=True, shadow=True, borderpad=1, labelspacing=.5,
               facecolor='lightblue', edgecolor=Constants.color_black)

    # Show plot
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define a set of professional and distinct colors
# These are often used in scientific publications due to their clarity and contrast.
# You can customize this palette further if you have specific journal requirements.
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']
# Blue, Orange, Green, Purple, Yellow, Light Blue, Gold - a "Wong" palette derivative

# Define distinct markers and linestyles for clear differentiation
MARKERS = ['o', 's', '^', 'D', 'v', 'X', 'P']  # Circle, Square, Up Triangle, Diamond, Down Triangle, X, Plus
# LINESTYLES = ['-', '--', '-.', ':']  # Solid, Dashed, Dash-dot, Dotted
LINESTYLES = [
    '-',          # 0: Solid
    '--',         # 1: Dashed
    '-.',         # 2: Dash-dot
    ':',          # 3: Dotted
    (0, (1, 10)), # 4: Loosely dotted (small dot, large gap)
    (0, (5, 1)),  # 5: Densely dashed (long dash, small gap)
    (0, (3, 10, 1, 10)), # 6: Loosely dash-dot (medium dash, large gap, small dot, large gap)
    (0, (8, 4, 2, 4)), # 7: Long dash, short dash (long dash, medium gap, short dash, medium gap)
    (0, (1, 2, 3, 2)), # 8: Dot-dash-dot (dot, small gap, medium dash, small gap)
    (0, (1, 1, 1, 5)), # 9: Alternating dots (dot, tiny gap, dot, larger gap)
]

# def plot_minibatches_results_standard_all2(kpi, x_axis_dao_gp, y_axis_dao_gp, label_dao_gp,
#                                                 x_axis_fitc, y_axis_fitc, label_fitc,
#                                                 x_axis_kpa, y_axis_kpa, label_kpa,
#                                                 legend_loc):
#     plt.style.use('seaborn-v0_8-whitegrid')
#     plt.figure(figsize=(8, 6))  # A standard size for papers (e.g., 8x6 inches)
#
#     # Plotting each series with distinct colors, markers, and linestyles
#     # Increased markersize and linewidth for better visibility
#     # Zorder ensures lines drawn last appear on top if they overlap
#     line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp,
#                       linestyle=LINESTYLES[0],
#                       marker=MARKERS[0],
#                       markersize=4,  # Larger markers
#                       linewidth=1.5,  # Thicker lines
#                       color=COLORS[0],
#                       label=label_dao_gp,
#                       zorder=10)  # Highest zorder
#
#     line2, = plt.plot(x_axis_fitc, y_axis_fitc,
#                       linestyle=LINESTYLES[1],
#                       marker=MARKERS[1],
#                       markersize=4,
#                       linewidth=1.5,
#                       color=COLORS[1],
#                       label=label_fitc,
#                       zorder=9)
#
#     line5, = plt.plot(x_axis_kpa, y_axis_kpa,  # Renamed from line3 for consistency with original example
#                       linestyle=LINESTYLES[2],
#                       marker=MARKERS[2],
#                       markersize=4,
#                       linewidth=1.5,
#                       color=COLORS[2],
#                       label=label_kpa,
#                       zorder=8)
#
#     # Adding labels and title with professional font sizes
#     # Use LaTeX for variable names for a cleaner look
#     plt.xlabel('Number of Samples ($N$)', fontsize=14, weight='bold')  # Clearer label
#
#     if kpi == 'R2':
#         plt.ylabel('$R^2$', fontsize=12, weight='bold')
#         title_kpi = 'Coefficient of Determination'
#     elif kpi == 'MSE':
#         plt.ylabel('MSE', fontsize=12, weight='bold')
#         title_kpi = 'Mean Squared Error'
#     else:
#         plt.ylabel(kpi, fontsize=12, weight='bold')  # Fallback if kpi is not R2/MSE
#         title_kpi = f'{kpi} Performance'
#
#     plt.title(f'{title_kpi} Comparison Across Models', fontsize=14, weight='bold', pad=15)  # Descriptive title
#
#     # Adjust font size of numbers on x-axis and y-axis ticks
#     plt.tick_params(axis='x', labelsize=12)
#     plt.tick_params(axis='y', labelsize=12)
#
#     # Customize grid for a subtle and professional look
#     # Grid lines are thinner and lighter
#     plt.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=0.7)
#
#     # Remove top and right spines for a clean, open look
#     sns.despine(top=True, right=True, left=False, bottom=False)
#
#     # Customize legend for professional appearance
#     plt.legend(fontsize=12, loc=legend_loc,
#                fancybox=True, shadow=True,
#                borderpad=0.8, labelspacing=0.6,
#                facecolor='white', edgecolor='lightgray', framealpha=0.9)
#
#     # Ensure all elements fit within the figure area, crucial for saving
#     plt.tight_layout()
#
#     # Show plot
#     plt.show()



#############################################

def plot_minibatches_results_standard_all2(kpi,
                                             x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                             x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                             x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                             X_test, y_test, # X_test and y_    test are passed but not directly plotted
                                             legend_loc):

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Prepare labels to include the final R2 score, formatted to 2 decimal places
    # Only append R2 if the KPI is 'R2' for consistency.
    # Otherwise, you might want to consider adding MSE for MSE plots.
    if kpi == 'R2':
        final_label_dao_gp = f"{label_dao_gp} (Test $R^2$: {dao_dp_final_r2:.2f})"
        final_label_fitc = f"{label_fitc} (Test $R^2$: {fitc_final_r2:.2f})"
        final_label_kpa = f"{label_kpa} (Test $R^2$: {kpa_r2:.2f})"
    elif kpi == 'MSE':
        # You might want to display final MSE here, if available
        # For this example, I'll assume you only want R2 in the legend for R2 plots
        # If you pass a final_mse for MSE plots, you can add it here similarly
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa
    else:
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa

    # Plotting each series with distinct colors, markers, and linestyles
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp,
                      linestyle=LINESTYLES[7],
                      marker=MARKERS[5],
                      markersize=4,  # Increased marker size slightly
                      linewidth=1.5,   # Increased line width slightly
                      color=COLORS[0],
                      label=final_label_dao_gp, # Use the updated label
                      zorder=10)

    line2, = plt.plot(x_axis_fitc, y_axis_fitc,
                      linestyle=LINESTYLES[1],
                      marker=MARKERS[1],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[1],
                      label=final_label_fitc, # Use the updated label
                      zorder=9)

    line5, = plt.plot(x_axis_kpa, y_axis_kpa,
                      linestyle=LINESTYLES[3],
                      marker=MARKERS[2],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[2],
                      label=final_label_kpa, # Use the updated label
                      zorder=8)

    # Adding labels and title with professional font sizes
    plt.xlabel('$N$', fontsize=12, weight='bold')

    if kpi == 'R2':
        plt.ylabel('$R^2$', fontsize=12, weight='bold')
        title_kpi = 'Coefficient of Determination'
    elif kpi == 'MSE':
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14, weight='bold')
        title_kpi = 'Mean Squared Error'
    else:
        plt.ylabel(kpi, fontsize=14, weight='bold')
        title_kpi = f'{kpi} Performance'

    plt.title(f'{title_kpi} Comparison Across Models', fontsize=14, weight='bold', pad=15)

    # Adjust font size of numbers on x-axis and y-axis ticks
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # Customize grid for a subtle and professional look
    plt.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=0.7)

    # Remove top and right spines for a clean, open look
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Customize legend for professional appearance
    legend = plt.legend(fontsize=12, loc=legend_loc,
               fancybox=True, shadow=True,
               borderpad=0.8, labelspacing=0.6,
               facecolor='white', edgecolor='lightgray', framealpha=0.9)
    legend.set_zorder(99)
    # Ensure all elements fit within the figure area, crucial for saving
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_minibatches_results_standard_all_abrupt(kpi,
                                                 x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                 x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                 x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                 X_test, y_test,
                                                 # X_test and y_    test are passed but not directly plotted
                                                 legend_loc, start_of_concept2):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Prepare labels to include the final R2 score, formatted to 2 decimal places
    # Only append R2 if the KPI is 'R2' for consistency.
    # Otherwise, you might want to consider adding MSE for MSE plots.
    if kpi == 'R2':
        final_label_dao_gp = f"{label_dao_gp} (Test $R^2$: {dao_dp_final_r2:.2f})"
        final_label_fitc = f"{label_fitc} (Test $R^2$: {fitc_final_r2:.2f})"
        final_label_kpa = f"{label_kpa} (Test $R^2$: {kpa_r2:.2f})"
    elif kpi == 'MSE':
        # You might want to display final MSE here, if available
        # For this example, I'll assume you only want R2 in the legend for R2 plots
        # If you pass a final_mse for MSE plots, you can add it here similarly
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa
    else:
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa

    # Plotting each series with distinct colors, markers, and linestyles
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp,
                      linestyle=LINESTYLES[7],
                      marker=MARKERS[5],
                      markersize=4,  # Increased marker size slightly
                      linewidth=1.5,  # Increased line width slightly
                      color=COLORS[0],
                      label=final_label_dao_gp  # Use the updated label
                      )

    line2, = plt.plot(x_axis_fitc, y_axis_fitc,
                      linestyle=LINESTYLES[1],
                      marker=MARKERS[1],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[1],
                      label=final_label_fitc  # Use the updated label
                      )

    line5, = plt.plot(x_axis_kpa, y_axis_kpa,
                      linestyle=LINESTYLES[3],
                      marker=MARKERS[2],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[2],
                      label=final_label_kpa  # Use the updated label
                      )

    # Add yellow shaded region for Concept 2
    end_of_concept = np.max(np.concatenate([x_axis_dao_gp, x_axis_fitc, x_axis_kpa]))
    shaded_patch = plt.axvspan(start_of_concept2, end_of_concept, color='palegoldenrod', alpha=0.3, label='Abrupt Drift',
                               zorder=1)

    # Adding labels and title with professional font sizes
    plt.xlabel('$N$', fontsize=12, weight='bold')

    if kpi == 'R2':
        plt.ylabel('$R^2$', fontsize=12, weight='bold')
        title_kpi = 'Coefficient of Determination'
    elif kpi == 'MSE':
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14, weight='bold')
        title_kpi = 'Mean Squared Error'
    else:
        plt.ylabel(kpi, fontsize=14, weight='bold')
        title_kpi = f'{kpi} Performance'

    plt.title(f'{title_kpi} Comparison Across Models', fontsize=14, weight='bold', pad=15)

    # Adjust font size of numbers on x-axis and y-axis ticks
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # Customize grid for a subtle and professional look
    plt.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=0.7)

    # Remove top and right spines for a clean, open look
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Customize legend for professional appearance
    legend = plt.legend(fontsize=12, loc=legend_loc,
                        fancybox=True, shadow=True,
                        borderpad=0.8, labelspacing=0.6,
                        facecolor='white', edgecolor='lightgray', framealpha=0.9)
    legend.set_zorder(100)
    # Ensure all elements fit within the figure area, crucial for saving
    plt.tight_layout()

    # Show plot
    plt.show()



def plot_minibatches_results_standard_all_incremental(kpi,
                                                 x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                 x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                 x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                 X_test, y_test,
                                                 # X_test and y_    test are passed but not directly plotted
                                                 legend_loc, drift_locations_every):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Prepare labels to include the final R2 score, formatted to 2 decimal places
    # Only append R2 if the KPI is 'R2' for consistency.
    # Otherwise, you might want to consider adding MSE for MSE plots.
    if kpi == 'R2':
        final_label_dao_gp = f"{label_dao_gp} (Test $R^2$: {dao_dp_final_r2:.2f})"
        final_label_fitc = f"{label_fitc} (Test $R^2$: {fitc_final_r2:.2f})"
        final_label_kpa = f"{label_kpa} (Test $R^2$: {kpa_r2:.2f})"
    elif kpi == 'MSE':
        # You might want to display final MSE here, if available
        # For this example, I'll assume you only want R2 in the legend for R2 plots
        # If you pass a final_mse for MSE plots, you can add it here similarly
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa
    else:
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa

    # Plotting each series with distinct colors, markers, and linestyles
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp,
                      linestyle=LINESTYLES[7],
                      marker=MARKERS[5],
                      markersize=4,  # Increased marker size slightly
                      linewidth=1.5,  # Increased line width slightly
                      color=COLORS[0],
                      label=final_label_dao_gp  # Use the updated label
                      )

    line2, = plt.plot(x_axis_fitc, y_axis_fitc,
                      linestyle=LINESTYLES[1],
                      marker=MARKERS[1],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[1],
                      label=final_label_fitc  # Use the updated label
                      )

    line5, = plt.plot(x_axis_kpa, y_axis_kpa,
                      linestyle=LINESTYLES[3],
                      marker=MARKERS[2],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[2],
                      label=final_label_kpa  # Use the updated label
                      )

    # Add yellow vertical lines on start of each new concept
    end_of_concept = int(np.max(np.concatenate([x_axis_dao_gp, x_axis_fitc, x_axis_kpa])))
    # Plot vertical yellow lines every `drift_locations_every` points
    for loc in range(drift_locations_every, end_of_concept + 1, drift_locations_every):
        plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6, label=None)
    # add to legend once
    first = True
    for loc in range(drift_locations_every, end_of_concept + 1, drift_locations_every):
        plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6,
                    label='Drift Start' if first else None)
        first = False

    # Adding labels and title with professional font sizes
    plt.xlabel('$N$', fontsize=12, weight='bold')

    if kpi == 'R2':
        plt.ylabel('$R^2$', fontsize=12, weight='bold')
        title_kpi = 'Coefficient of Determination'
    elif kpi == 'MSE':
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14, weight='bold')
        title_kpi = 'Mean Squared Error'
    else:
        plt.ylabel(kpi, fontsize=14, weight='bold')
        title_kpi = f'{kpi} Performance'

    plt.title(f'{title_kpi} Comparison Across Models', fontsize=14, weight='bold', pad=15)

    # Adjust font size of numbers on x-axis and y-axis ticks
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # Customize grid for a subtle and professional look
    plt.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=0.7)

    # Remove top and right spines for a clean, open look
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Customize legend for professional appearance
    legend = plt.legend(fontsize=12, loc=legend_loc,
                        fancybox=True, shadow=True,
                        borderpad=0.8, labelspacing=0.6,
                        facecolor='white', edgecolor='lightgray', framealpha=0.9)
    legend.set_zorder(100)
    # Ensure all elements fit within the figure area, crucial for saving
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_minibatches_results_standard_all_gradual(kpi,
                                                 x_axis_dao_gp, y_axis_dao_gp, label_dao_gp, dao_dp_final_r2,
                                                 x_axis_fitc, y_axis_fitc, label_fitc, fitc_final_r2,
                                                 x_axis_kpa, y_axis_kpa, label_kpa, kpa_r2,
                                                 X_test, y_test,
                                                 # X_test and y_    test are passed but not directly plotted
                                                 legend_loc,
                                                 gradual_drift_locations,
                                                 gradual_drift_concepts ):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Prepare labels to include the final R2 score, formatted to 2 decimal places
    # Only append R2 if the KPI is 'R2' for consistency.
    # Otherwise, you might want to consider adding MSE for MSE plots.
    if kpi == 'R2':
        final_label_dao_gp = f"{label_dao_gp} (Test $R^2$: {dao_dp_final_r2:.2f})"
        final_label_fitc = f"{label_fitc} (Test $R^2$: {fitc_final_r2:.2f})"
        final_label_kpa = f"{label_kpa} (Test $R^2$: {kpa_r2:.2f})"
    elif kpi == 'MSE':
        # You might want to display final MSE here, if available
        # For this example, I'll assume you only want R2 in the legend for R2 plots
        # If you pass a final_mse for MSE plots, you can add it here similarly
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa
    else:
        final_label_dao_gp = label_dao_gp
        final_label_fitc = label_fitc
        final_label_kpa = label_kpa

    # Plotting each series with distinct colors, markers, and linestyles
    line1, = plt.plot(x_axis_dao_gp, y_axis_dao_gp,
                      linestyle=LINESTYLES[7],
                      marker=MARKERS[5],
                      markersize=4,  # Increased marker size slightly
                      linewidth=1.5,  # Increased line width slightly
                      color=COLORS[0],
                      label=final_label_dao_gp  # Use the updated label
                      )

    line2, = plt.plot(x_axis_fitc, y_axis_fitc,
                      linestyle=LINESTYLES[1],
                      marker=MARKERS[1],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[1],
                      label=final_label_fitc  # Use the updated label
                      )

    line5, = plt.plot(x_axis_kpa, y_axis_kpa,
                      linestyle=LINESTYLES[3],
                      marker=MARKERS[2],
                      markersize=4,
                      linewidth=1.5,
                      color=COLORS[2],
                      label=final_label_kpa  # Use the updated label
                      )

    from matplotlib.patches import Patch

    # Define concept colors
    concept_colors = {
        'c1': Constants.color_light_green,  # c1 starts with green
        'c2': Constants.color_light_yellow  # c2 ends with yellow
    }

    # Define region boundaries
    region_starts = [0] + gradual_drift_locations[:-1]
    region_ends = gradual_drift_locations

    # Plot all shaded concept regions
    for start, end, concept in zip(region_starts, region_ends, gradual_drift_concepts):
        color = concept_colors.get(concept, Constants.color_blue)
        plt.axvspan(start, end, color=color, alpha=0.3, label=None)

    # Only show legend entries for first and last concepts
    first_concept = gradual_drift_concepts[0]
    last_concept = gradual_drift_concepts[-1]

    custom_legend_handles = [
        Patch(facecolor=concept_colors[first_concept], edgecolor='none', alpha=0.3,
              label=f'Start Concept: {first_concept.upper()}'),
        Patch(facecolor=concept_colors[last_concept], edgecolor='none', alpha=0.3,
              label=f'End Concept: {last_concept.upper()}')
    ]

    plt.legend(handles=custom_legend_handles, loc=legend_loc, fontsize=12)

    # Adding labels and title with professional font sizes
    plt.xlabel('$N$', fontsize=12, weight='bold')

    if kpi == 'R2':
        plt.ylabel('$R^2$', fontsize=12, weight='bold')
        title_kpi = 'Coefficient of Determination'
    elif kpi == 'MSE':
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14, weight='bold')
        title_kpi = 'Mean Squared Error'
    else:
        plt.ylabel(kpi, fontsize=14, weight='bold')
        title_kpi = f'{kpi} Performance'

    plt.title(f'{title_kpi} Comparison Across Models', fontsize=14, weight='bold', pad=15)

    # Adjust font size of numbers on x-axis and y-axis ticks
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # Customize grid for a subtle and professional look
    plt.grid(True, linestyle=':', alpha=0.6, color='gray', linewidth=0.7)

    # Remove top and right spines for a clean, open look
    sns.despine(top=True, right=True, left=False, bottom=False)
    # Combine all legend handles
    concept_handles = [
        Patch(facecolor=concept_colors[first_concept], edgecolor='none', alpha=0.3,
              label=f'Start Concept: {first_concept}'),
        Patch(facecolor=concept_colors[last_concept], edgecolor='none', alpha=0.3,
              label=f'End Concept: {last_concept}')
    ]
    model_handles, model_labels = plt.gca().get_legend_handles_labels()
    all_handles = model_handles + concept_handles
    # Customize legend for professional appearance
    legend = plt.legend(handles=all_handles,fontsize=12, loc=legend_loc,
                        fancybox=True, shadow=True,
                        borderpad=0.8, labelspacing=0.6,
                        facecolor='white', edgecolor='lightgray', framealpha=0.9)
    legend.set_zorder(100)
    # Ensure all elements fit within the figure area, crucial for saving
    plt.tight_layout()

    # Show plot
    plt.show()