from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask_cors import CORS
import requests
from io import StringIO, BytesIO

app = Flask(__name__)
CORS(app)

# Features supported for prediction
features = [
    'coal_extracted(tons)',
    'fuel_used(liters)',
    'electricity_used(kwh)',
    'fuel_emission',
    'electricity_emission',
    'total_emission',
    'methane_emission(m3)'
]

# Forecast future values using linear regression
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
        year = int(request.args.get('year'))
        factor_name = request.args.get('factor_name')
        mine_name = request.args.get('mine_name')

        if year < 2024 or year > 2033:
            return jsonify({"error": "Invalid year. Please provide a year between 2024 and 2033."}), 400

        if factor_name not in features:
            return jsonify({"error": "Invalid factor_name. Please provide a valid factor name."}), 400

        # RAW GitHub URLs for the CSV and PKL files
        base_url = "https://raw.githubusercontent.com/eswarsainandan04/mines-backend/main"
        data_url = f"{base_url}/{mine_name}_data.csv"
        model_url = f"{base_url}/{factor_name}_{mine_name}.pkl"

        # Load CSV from GitHub
        data_response = requests.get(data_url)
        if data_response.status_code != 200:
            return jsonify({"error": f"Data not found for mine: {mine_name}"}), 404
        data = pd.read_csv(StringIO(data_response.text))

        # Load PKL from GitHub
        model_response = requests.get(model_url)
        if model_response.status_code != 200:
            return jsonify({"error": f"Model not found for factor: {factor_name} and mine: {mine_name}"}), 404
        model = joblib.load(BytesIO(model_response.content))

        # Predict using historical projection
        projected_values = project_future_values(data, factor_name, [year])
        prediction = float(projected_values[0])

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
