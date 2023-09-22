import pandas as pd
from sklearn.linear_model import LinearRegression
model= pd.read_pickle('infosys.pkl')


today_features = {
    'Adj Close': 1496.00,  # yesterday's closing price   
    'High': 1506.70,    
    'Low': 1485.90,    #low value for today
    'Volume': 411402,
    'Open': 1494.35 # Example trading volume for today
}

# Create a DataFrame with the input data for today
today_data = pd.DataFrame([today_features])

# Make a prediction for today's closing value
predicted_closing_value = model.predict(today_data)

# Print the predicted closing value for today
print("Predicted Closing Value for Today:", predicted_closing_value[0])