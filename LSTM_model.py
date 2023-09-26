import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') ## use this if we want to save the figures in file explorer.
import numpy as np
import pandas_ta as ta
# Define the stock symbol and date range
symbol = "INFY.BO"  # Example: Apple Inc.
start_date = "2000-02-08"
end_date = "2023-09-21"

# Fetch historical data
data = yf.download(symbol, start=start_date, end=end_date)
data = pd.read_csv('INFY.BO.csv')
print(data.head())

# Check the first few rows of the dataset
print(data.head())

# Get information about the dataset
print(data.info())

# Summary statistics
print(data.describe())

# Data Visualization
# Example: Plotting stock prices over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.title('Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('close_value.png')
# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Include relevant numeric data types

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
plt.savefig('heatmap.png')

#####################################stage 2################################
##stage2 feature engineering
data['SMA_50'] = ta.sma(data['Close'], length=50)
data['EMA_50'] = ta.ema(data['Close'], length=50)

# Calculate Relative Strength Index (RSI)
data['RSI_14'] = ta.rsi(data['Close'], length=14)

# Calculate Bollinger Bands
data.ta.bbands(length=20, append=True)

# Calculate daily returns
data['Daily_Return'] = data['Close'].pct_change() * 100

# Calculate basic statistics
mean_return = data['Daily_Return'].mean()
std_return = data['Daily_Return'].std()
max_return = data['Daily_Return'].max()
min_return = data['Daily_Return'].min()

print("Mean Daily Return:", mean_return)
print("Standard Deviation of Daily Return:", std_return)
print("Maximum Daily Return:", max_return)
print("Minimum Daily Return:", min_return)

# Display the updated DataFrame with added features
print(data.head())
# Plot the daily returns
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Daily_Return'], label='Daily Return', color='green')
plt.title('Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('daily_return.png')
print(data.head())
data_cleaned = data.dropna(axis=1)
data_cleaned = data.dropna()
print(data_cleaned.head(100))
#data_cleaned = data.drop(data.columns[7], axis=1)
data= data_cleaned
# print(data.head())


####################LSTM model###############################################

# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Define the features and target variable
# Define the features and target variable
features = ['Adj Close', 'Daily_Return', 'Low', 'Volume']  # Include relevant features
target = 'High'

# Extract the relevant data
data = data[features + [target]]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Define a function to create sequences from the data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])
        y.append(data[i+sequence_length, -1])
    return np.array(X), np.array(y)

# Create sequences for training and test sets
sequence_length = 10  # You can adjust this based on your needs
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Define the LSTM model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Evaluate the model
y_pred = model.predict(X_test)

# Inverse transform the predicted values and actual values
y_pred = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual High')
plt.plot(y_pred, label='Predicted High')
plt.legend()
plt.xlabel('Time')
plt.ylabel('High Price')
plt.show()
import pickle
# Save the model
pickle.dump(model, open('infosysLSTM.pkl', 'wb'))