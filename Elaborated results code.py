import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)

# Metrics calculations
mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_test, y_pred_xgb)
mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
r2 = r2_score(y_test, y_pred_xgb)
explained_var = explained_variance_score(y_test, y_pred_xgb)

# Correlation coefficient (Pearson's r)
correlation = np.corrcoef(y_test, y_pred_xgb)[0, 1]

# Mean Absolute Scaled Error (MASE)
def mean_absolute_scaled_error(y_true, y_pred):
    naive_forecast = np.roll(y_true, 1)  # Shift actuals as a naive forecast
    naive_error = np.mean(np.abs(y_true[1:] - naive_forecast[1:]))
    scaled_error = np.mean(np.abs(y_true - y_pred)) / naive_error
    return scaled_error

mase = mean_absolute_scaled_error(y_test.values, y_pred_xgb)

# Print metrics
metrics = {
    "Mean Absolute Error (MAE)": mae,
    "Mean Squared Error (MSE)": mse,
    "Root Mean Squared Error (RMSE)": rmse,
    "Median Absolute Error (MedAE)": medae,
    "Mean Absolute Percentage Error (MAPE)": f"{mape:.2f}%",
    "R-squared (R²)": r2,
    "Explained Variance Score": explained_var,
    "Correlation Coefficient (Pearson's r)": correlation,
    "Mean Absolute Scaled Error (MASE)": mase,
}

print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# Residuals
residuals = y_test - y_pred_xgb

# Visualizations
plt.figure(figsize=(18, 12))

# Subplot 1: Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_xgb, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Actual Stock Price")
plt.ylabel("Predicted Stock Price")
plt.grid(True)

# Subplot 2: Residual Distribution
plt.subplot(2, 2, 2)
plt.hist(residuals, bins=30, color='purple', alpha=0.7)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)

# Subplot 3: Residuals vs Predicted
plt.subplot(2, 2, 3)
plt.scatter(y_pred_xgb, residuals, alpha=0.7, color='orange')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Residuals vs Predicted Stock Prices")
plt.xlabel("Predicted Stock Price")
plt.ylabel("Residuals")
plt.grid(True)

# Subplot 4: Actual, Predicted, and Sentiment (Random 250 points)
sampled_test_data = test_data.sample(n=500, random_state=42).sort_index()
plt.subplot(2, 2, 4)
plt.plot(sampled_test_data.index, sampled_test_data['stock_price'], label='Actual Stock Price', color='green')
plt.plot(sampled_test_data.index, sampled_test_data['Predicted Stock Price'], label='Predicted Stock Price', color='red')
plt.plot(sampled_test_data.index, sampled_test_data['Sentiment Score'], label='Sentiment Score', color='blue')
plt.title("Actual, Predicted, and Sentiment (250 Points)")
plt.xlabel("Row Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()