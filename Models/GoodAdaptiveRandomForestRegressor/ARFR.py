import numpy as np
from river import ensemble, preprocessing, metrics, compose
from river import stream
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def generate_data(n_samples=2000, noise=0.2):
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10 - 5  # X in [-5, 5]
    y = np.sin(X[:, 0]) + np.random.randn(n_samples) * noise
    return X, y

# Convert to River compatible format (dictionaries)
def numpy_to_dict(X):
    return [{'x': float(x[0])} for x in X]

def arfr(train_data):
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            lambda_value=6,
            seed=42,
            drift_detector=None,
            warning_detector=None,
            metric=metrics.RMSE()
        )
    )

    metric = metrics.RMSE()
    training_r2 = []
    training_mse = []

    print("Training model incrementally...")
    for i, (xi, yi) in enumerate(train_data, 1):
        # Make prediction before learning (to test on unseen data)
        y_pred = model.predict_one(xi)

        # Update model and metrics
        model.learn_one(xi, yi)
        metric.update(yi, y_pred)

        # Calculate R² and MSE periodically
        if i % 100 == 0:
            current_r2 = r2_score(y_train[:i], [model.predict_one(x) for x, _ in train_data[:i]])
            current_mse = mean_squared_error(y_train[:i], [model.predict_one(x) for x, _ in train_data[:i]])
            training_r2.append(current_r2)
            training_mse.append(current_mse)
            print(
                f"Processed {i}/{len(train_data)} samples | RMSE: {metric.get():.4f} | R²: {current_r2:.4f} | MSE: {current_mse:.4f}")
            metric = metrics.RMSE()  # Reset metric
    return model

if __name__ == "__main__":
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_data = list(zip(numpy_to_dict(X_train), y_train))
    test_data = list(zip(numpy_to_dict(X_test), y_test))

    model = arfr(train_data)
    y_pred_test = [model.predict_one(x) for x, _ in test_data]
    final_r2 = r2_score(y_test, y_pred_test)
    final_mse = mean_squared_error(y_test, y_pred_test)

    print("\n=== Final Test Performance ===")
    print(f"R²: {final_r2:.4f}")
    print(f"MSE: {final_mse:.4f}")



