import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') ## use this if we want to save the figures in file explorer.
import numpy as np
import pandas_ta as ta
# Define the stock symbol and date range
symbol = "IRFC.NS"  # Example: Apple Inc.
start_date = "2021-02-08"
end_date = "2023-09-21"

# Fetch historical data
#data = yf.download(symbol, start=start_date, end=end_date)
data = pd.read_csv('data.csv')
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

###############################stage 3##############################
###ML model####