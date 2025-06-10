import numpy as np

def evaluate_cocept_drift(conceptDriftDetector, memoryManager, KPI, z, min_window_length=7, max_window_length=31, safe_threshold=0 ):
    is_drift = None
    drift_type = None
    if len(memoryManager.mini_batch_data) >= min_window_length:
        KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data, KPI, max_window_length)
        print("KPI_Window_ST", KPI_Window_ST)
        threshold, mean_kpi, std_kpi, lower_higher_limit, drift_magnitude = conceptDriftDetector.get_meaures(KPI_Window_ST, z, KPI)
        print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1], "lower_higher_limit", lower_higher_limit, "drift_magnitude", drift_magnitude)
        is_drift, drift_type = conceptDriftDetector.drift_classifier(KPI_Window_ST, mean_kpi, lower_higher_limit, KPI, safe_threshold)
        print("is_drit", is_drift, "drift_type", drift_type)
    return is_drift, drift_type
