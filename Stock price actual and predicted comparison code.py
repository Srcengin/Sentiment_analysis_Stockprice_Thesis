import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Randomly sample 500 points for visualization
sampled_test_data = test_data.sample(n=500, random_state=42).sort_index()

# Normalize the values for better comparison
scaler = MinMaxScaler()
sampled_test_data['Normalized Sentiment Score'] = scaler.fit_transform(sampled_test_data[['Sentiment Score']])
sampled_test_data['Normalized Stock Price'] = scaler.fit_transform(sampled_test_data[['stock_price']])
sampled_test_data['Normalized Predicted Stock Price'] = scaler.fit_transform(sampled_test_data[['Predicted Stock Price']])

# Apply a rolling average to smooth the lines
sampled_test_data['Smoothed Sentiment Score'] = sampled_test_data['Normalized Sentiment Score'].rolling(window=10).mean()
sampled_test_data['Smoothed Stock Price'] = sampled_test_data['Normalized Stock Price'].rolling(window=10).mean()
sampled_test_data['Smoothed Predicted Stock Price'] = sampled_test_data['Normalized Predicted Stock Price'].rolling(window=10).mean()

# Plot the smoothed data
plt.figure(figsize=(14, 8))

# Plot smoothed sentiment score
plt.plot(sampled_test_data.index, sampled_test_data['Smoothed Sentiment Score'], label='Sentiment Score', color='blue', linestyle='-', alpha=0.8)

# Plot smoothed actual stock prices
plt.plot(sampled_test_data.index, sampled_test_data['Smoothed Stock Price'], label='Actual Stock Price', color='green', linestyle='-', alpha=0.8)

# Plot smoothed predicted stock prices
plt.plot(sampled_test_data.index, sampled_test_data['Smoothed Predicted Stock Price'], label='Predicted Stock Price', color='red', linestyle='-', alpha=0.8)

# Add labels, title, and legend
plt.title('Comparison of Sentiment Score, Actual Stock Price, and Predicted Stock Price (Smoothed)', fontsize=16)
plt.xlabel('Row Index (Sampled)', fontsize=12)
plt.ylabel('Normalized Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Show the plot
plt.show()
