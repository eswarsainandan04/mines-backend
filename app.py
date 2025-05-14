from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Features (assuming they are consistent across all mines)
features = ['coal_extracted(tons)', 'fuel_used(liters)', 'electricity_used(kwh)','fuel_emission','electricity_emission', 'total_emission','methane_emission(m3)']

# Project future values using historical trends (unchanged)
def project_future_values(data, feature_name, future_years):
    model = LinearRegression()
    years = data['Year'].values.reshape(-1, 1)
    feature_values = data[feature_name].values
    model.fit(years, feature_values)

    future_years_array = np.array(future_years).reshape(-1, 1)
    projected_values = model.predict(future_years_array)

    return projected_values

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the year, factor_name, and mine_name from the query parameters
        year = int(request.args.get('year'))
        factor_name = request.args.get('factor_name')
        mine_name = request.args.get('mine_name')

        # Validate the year
        if year < 2024 or year > 2033:
            return jsonify({"error": "Invalid year. Please provide a year between 2024 and 2033."}), 400

        # Validate the factor_name
        if factor_name not in features:
            return jsonify({"error": "Invalid factor_name. Please provide a valid factor name."}), 400

        # Construct the data and model file paths based on mine_name
        data_file_path = f'{mine_name}_data.csv'
        model_file_path = f'{factor_name}_{mine_name}.pkl'

        # Check if the data and model files exist
        if not os.path.exists(data_file_path):
            return jsonify({"error": f"Data not found for mine: {mine_name}"}), 404
        if not os.path.exists(model_file_path):
            return jsonify({"error": f"Model not found for factor: {factor_name} and mine: {mine_name}"}), 404

        # Load the data and model
        data = pd.read_csv(data_file_path)
        model = joblib.load(model_file_path)

        # Make prediction 
        projected_values = project_future_values(data, factor_name, [year])
        prediction = projected_values[0]

        return jsonify({
            "mine_name": mine_name,
            "year": year,
            "factor_name": factor_name,
            "prediction": prediction
        })

    except ValueError:
        return jsonify({"error": "Invalid year format. Please provide an integer."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
