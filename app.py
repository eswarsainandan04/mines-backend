from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Features expected for predictions
features = [
    'coal_extracted(tons)', 'fuel_used(liters)', 'electricity_used(kwh)',
    'fuel_emission', 'electricity_emission', 'total_emission', 'methane_emission(m3)'
]

def project_future_values(data, feature_name, future_years):
    model = LinearRegression()
    years = data['Year'].values.reshape(-1, 1)
    feature_values = data[feature_name].values
    model.fit(years, feature_values)
    future_years_array = np.array(future_years).reshape(-1, 1)
    return model.predict(future_years_array)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract query parameters
        year_str = request.args.get('year')
        factor_name = request.args.get('factor_name')
        mine_name = request.args.get('mine_name')

        # Validate parameters
        if not year_str:
            return jsonify({"error": "Missing 'year' query parameter."}), 400
        if not factor_name:
            return jsonify({"error": "Missing 'factor_name' query parameter."}), 400
        if not mine_name:
            return jsonify({"error": "Missing 'mine_name' query parameter."}), 400

        # Parse year
        try:
            year = int(year_str)
        except ValueError:
            return jsonify({"error": "Invalid year format. Please provide an integer."}), 400

        if year < 2024 or year > 2033:
            return jsonify({"error": "Invalid year. Please provide a year between 2024 and 2033."}), 400

        if factor_name not in features:
            return jsonify({"error": f"Invalid factor_name. Choose from: {', '.join(features)}"}), 400

        # File paths
        data_file_path = f'{mine_name}_data.csv'
        model_file_path = f'{factor_name}_{mine_name}.pkl'

        if not os.path.exists(data_file_path):
            return jsonify({"error": f"Data not found for mine: {mine_name}"}), 404
        if not os.path.exists(model_file_path):
            return jsonify({"error": f"Model not found for factor: {factor_name} and mine: {mine_name}"}), 404

        # Load data and model
        data = pd.read_csv(data_file_path)
        model = joblib.load(model_file_path)

        # Predict
        projected_values = project_future_values(data, factor_name, [year])
        prediction = float(projected_values[0])

        return jsonify({
            "mine_name": mine_name,
            "year": year,
            "factor_name": factor_name,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
