import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # Use joblib for model persistence

# Load historical data
data = pd.read_csv('lakhanpur_data.csv')

# Features and target variables
features = ['coal_extracted(tons)', 'fuel_used(liters)', 'electricity_used(kwh)','fuel_emission','electricity_emission', 'total_emission','methane_emission(m3)']

# Project future values using historical trends
def project_future_values(data, feature_name, future_years):
    model = LinearRegression()
    years = data['Year'].values.reshape(-1, 1)
    feature_values = data[feature_name].values
    model.fit(years, feature_values)

    future_years_array = np.array(future_years).reshape(-1, 1)
    projected_values = model.predict(future_years_array)

    return projected_values

# Define future years
future_years = np.arange(2024, 2034)

# Train a model for each feature and predict future values
models = {}  # Dictionary to store trained models
for feature in features:
    # Prepare the data
    X = data[['Year']]
    y = data[feature]

    # Split the data into training and testing sets (optional if you want to validate)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Gradient Boosting model
    model_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model_gb.fit(X_train, y_train)

    # Save the trained model using joblib
    joblib.dump(model_gb, f'{feature}_lakhanpur.pkl')

    # Store the model in the dictionary
    models[feature] = model_gb

print("Models trained and saved successfully!")