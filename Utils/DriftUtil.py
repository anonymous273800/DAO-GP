import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Dict, Union, List


def get_y_ks_p_value(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Computes the Kolmogorov-Smirnov test p-value between two target arrays.
    A low p-value (< 0.05) indicates significant distributional shift.
    """
    return ks_2samp(y1.ravel(), y2.ravel()).pvalue


def get_y_js_divergence(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Computes Jensen-Shannon Divergence between two target arrays.
    Value ranges from 0 (identical) to 1 (completely different).
    """
    combined = np.concatenate([y1.ravel(), y2.ravel()])
    bins = np.histogram_bin_edges(combined, bins=20)

    hist1, _ = np.histogram(y1.ravel(), bins=bins, density=True)
    hist2, _ = np.histogram(y2.ravel(), bins=bins, density=True)

    epsilon = 1e-10
    return float(jensenshannon(hist1 + epsilon, hist2 + epsilon))


def get_x_drift_metrics(X1: np.ndarray, X2: np.ndarray) -> Dict[str, Union[float, bool]]:
    """
    Computes drift metrics on input features:
    - Any KS p-value < 0.05?
    - Minimum KS p-value
    - Average JS divergence across features
    """
    results = {}
    if X1.shape[1] == 0 or X1.size == 0 or X2.size == 0:
        return {
            'features_any_ks_significant': False,
            'features_min_ks_p_value': np.nan,
            'features_avg_js_divergence': np.nan
        }

    num_features = X1.shape[1]
    feature_ks_p_values: List[float] = []
    feature_js_divergences: List[float] = []

    for i in range(num_features):
        x1_feat = X1[:, i].ravel()
        x2_feat = X2[:, i].ravel()

        # KS Test
        ks_p = ks_2samp(x1_feat, x2_feat).pvalue
        feature_ks_p_values.append(ks_p)

        # JS Divergence with shared binning
        combined = np.concatenate([x1_feat, x2_feat])
        bins = np.histogram_bin_edges(combined, bins=20)

        hist1, _ = np.histogram(x1_feat, bins=bins, density=True)
        hist2, _ = np.histogram(x2_feat, bins=bins, density=True)

        epsilon = 1e-10
        p = hist1 + epsilon
        q = hist2 + epsilon
        p /= np.sum(p)
        q /= np.sum(q)
        js_div = jensenshannon(p, q)
        feature_js_divergences.append(float(js_div))

    results['features_any_ks_significant'] = any(p < 0.05 for p in feature_ks_p_values)
    results['features_min_ks_p_value'] = float(np.min(feature_ks_p_values))
    results['features_avg_js_divergence'] = float(np.mean(feature_js_divergences))

    return results


def quantify_drift(X1, y1, X2, y2):
    y_ks_p_value = get_y_ks_p_value(y1, y2)
    y_js_divergence = get_y_js_divergence(y1, y2)
    x_drift_metrics = get_x_drift_metrics(X1, X2)

    print(f"{'y_ks_p_value':<25}: {y_ks_p_value:.4f}")
    print(f"{'y_js_divergence':<25}: {y_js_divergence:.4f}")
    print(f"{'features_any_ks_significant':<25}: {x_drift_metrics['features_any_ks_significant']}")
    print(f"{'features_min_ks_p_value':<25}: {x_drift_metrics['features_min_ks_p_value']:.4f}")
    print(f"{'features_avg_js_divergence':<25}: {x_drift_metrics['features_avg_js_divergence']:.4f}")
    print("-------------------------------------------------------------")

    return y_ks_p_value, y_js_divergence, x_drift_metrics


# --- Sample Dataset Functions ---
def DS007_ParabolicWave(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=5, weight_factor=0.5):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = X[:, 0] ** 2 * np.sin(stretch_factor * X[:, 0])
    if n_features > 1:
        y += weight_factor * X[:, 1:].sum(axis=1)
    y += noise * np.random.randn(n_samples)
    return X, y


def DS001_Sinusoidal_Not_Normalized(n_samples, n_features, noise, lower_bound, upper_bound, stretch_factor=1, y_shift=0.0):
    X = np.random.uniform(lower_bound, upper_bound, size=(n_samples, n_features))
    y = np.sin((2 * np.pi / stretch_factor) * X[:, 0]) + y_shift + noise * np.random.randn(n_samples)
    return X, y


# --- Example Use ---
if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 5000
    n_features = 50
    noise = 0.2

    # Concept 1
    X1, y1 = DS007_ParabolicWave(n_samples, n_features, noise, lower_bound=5, upper_bound=8)

    # Concept 2
    X2, y2 = DS001_Sinusoidal_Not_Normalized(n_samples, n_features, noise, lower_bound=-1, upper_bound=2, y_shift=50)

    quantify_drift(X1, y1, X2, y2)