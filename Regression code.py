import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from xgboost import XGBRegressor

# Step 1: Load Data from a Single CSV File
data = pd.read_csv("data.csv")  # Load the full dataset

# Step 2: Shuffle the data randomly
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle all rows

# Step 3: Encode 'Stock Name' Using Label Encoding
# Convert each stock name into a unique integer
label_encoder = LabelEncoder()
data['Stock Name'] = label_encoder.fit_transform(data['Stock Name'])

# Step 4: Split the data into 70% training and 30% testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Step 5: Extract Features and Target
X_train = train_data.drop(columns=['stock_price'])  # All columns except the target
y_train = train_data['stock_price']  # Target for training

X_test = test_data.drop(columns=['stock_price'])  # All columns except the target
y_test = test_data['stock_price']  # Target for testing

# Step 6: Scale Only the Numerical Features
# Identify numerical columns for scaling
numerical_columns = ['sentiment_score']  # Add more numerical columns if applicable
scaler = StandardScaler()

# Scale the numerical columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 7: Initialize and Train the XGBoost Regressor Model
model_xgb = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6)
model_xgb.fit(X_train, y_train)  # Train the model

# Step 8: Make Predictions on the Test Data
y_pred_xgb = model_xgb.predict(X_test)  # Predict on test data

# Step 9: Evaluate the Model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)  # Calculate Mean Absolute Error
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))  # Calculate Root Mean Squared Error

# Output the evaluation metrics
print(f'XGBoost Regression - Mean Absolute Error (MAE): {mae_xgb}')
print(f'XGBoost Regression - Root Mean Squared Error (RMSE): {rmse_xgb}')

# Output the predicted values (show a few for verification)
print("Predicted Stock Prices (Sample):", y_pred_xgb[:10])  # Print first 10 predictions
