from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('churn_model.joblib')
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        
        # Convert data to correct types and handle TotalCharges
        data['tenure'] = int(data['tenure'])
        data['MonthlyCharges'] = float(data['MonthlyCharges'])
        # A simple estimation for TotalCharges based on tenure for the demo
        data['TotalCharges'] = data['MonthlyCharges'] * data['tenure'] 
        data['SeniorCitizen'] = int(data['SeniorCitizen'])
        
        input_df = pd.DataFrame([data])
        
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        churn_prob_percent = probability[0][1] * 100
        
        output = {
            'prediction': 'Churn' if int(prediction[0]) == 1 else 'No Churn',
            'churn_probability': f"{churn_prob_percent:.2f}"
        }
        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)