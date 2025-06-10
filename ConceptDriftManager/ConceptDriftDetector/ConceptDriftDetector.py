import numpy as np
import math

class ConceptDriftDetector:

    def drift_classifier(self, KPI_Window_ST, mean_kpi, lower_higher_limit, kpi, safe_threshold=0):
        current_Window_KPI = KPI_Window_ST[-1]
        is_drift = None
        drift_type = None

        if kpi == 'R2':
            if current_Window_KPI < (mean_kpi - safe_threshold) and current_Window_KPI >= lower_higher_limit:
                is_drift = True
                drift_type = "ST"  # short term drift
            elif current_Window_KPI < (mean_kpi - safe_threshold) and current_Window_KPI < lower_higher_limit:
                is_drift = True
                drift_type = "LT"  # long term drift

        return is_drift, drift_type

    def get_KPI_Window_ST(self, mini_batch_data, KPI, window_length):
        KPI_Window = np.array([])
        mini_batch_data_amended = mini_batch_data[-window_length:]
        if KPI == 'R2':
            KPI_Window = np.concatenate([np.array([data.get_r2()]) for data in mini_batch_data_amended])
        elif KPI == 'MSE':
            KPI_Window = np.concatenate([np.array([data.get_cost()]) for data in mini_batch_data_amended])

        return KPI_Window

    def normalize_data(self, data):
        max_val = max(data)
        min_val = min(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    def get_meaures(self, KPI_Window, multiplier, kpi):
        KPI_Window = KPI_Window.astype(float)
        std_kpi = np.std(KPI_Window[:-1])
        if (math.isinf(std_kpi)): KPI_Window = self.normalize_data(KPI_Window)

        mean_kpi = np.mean(KPI_Window[:-1])
        std_kpi = np.std(KPI_Window[:-1])
        threshold = (multiplier * std_kpi)
        limit_deviated_kpi = 0

        if kpi == 'R2':
            lower_limit_deviated_kpi = mean_kpi - (multiplier * std_kpi)
            limit_deviated_kpi = lower_limit_deviated_kpi
            drift_magnitude = float("{:.5f}".format(mean_kpi - KPI_Window[-1])) if KPI_Window[-1] < mean_kpi else 0
        if kpi == 'MSE':
            higher_limit_deviated_kpi = mean_kpi + (multiplier * std_kpi)
            limit_deviated_kpi = higher_limit_deviated_kpi
            drift_magnitude = float("{:.5f}".format(KPI_Window[-1] - mean_kpi)) if KPI_Window[-1] > mean_kpi else 0

        return threshold, mean_kpi, std_kpi, limit_deviated_kpi, drift_magnitude