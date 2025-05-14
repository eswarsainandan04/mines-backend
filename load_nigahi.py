import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load historical data
data = pd.read_csv('nigahi_data.csv')

# Features
features = ['coal_extracted(tons)', 'fuel_used(liters)', 'electricity_used(kwh)','fuel_emission','electricity_emission', 'total_emission','methane_emission(m3)']

# Project future values using historical trends (same as before)
def project_future_values(data, feature_name, future_years):
    model = LinearRegression()
    years = data['Year'].values.reshape(-1, 1)
    feature_values = data[feature_name].values
    model.fit(years, feature_values)

    future_years_array = np.array(future_years).reshape(-1, 1)
    projected_values = model.predict(future_years_array)

    return projected_values
# ... (same implementation as in the training code)

# Define future years
future_years = np.arange(2024, 2034)

# Create a DataFrame to store future projections
future_df = pd.DataFrame({'Year': future_years})

# Load the trained models and make predictions
for feature in features:
    # Load the saved model
    model = joblib.load(f'{feature}_nigahi.pkl')

    # Project future values for the feature
    future_df[feature] = project_future_values(data, feature, future_years)

# Plot the future projections


print(future_df)