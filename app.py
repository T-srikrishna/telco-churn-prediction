from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

try:
    model = joblib.load('churn_model.joblib')
    # Extract feature names and coefficients for visualization
    feature_names = []
    # Add numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    feature_names.extend(numerical_cols)
    
    # Add categorical features (one-hot encoded)
    categorical_features = {
        'gender': ['Male'],
        'Partner': ['Yes'], 
        'Dependents': ['Yes'],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No phone service', 'Yes'],
        'InternetService': ['Fiber optic', 'No'],
        'OnlineSecurity': ['No internet service', 'Yes'],
        'OnlineBackup': ['No internet service', 'Yes'],
        'DeviceProtection': ['No internet service', 'Yes'],
        'TechSupport': ['No internet service', 'Yes'],
        'StreamingTV': ['No internet service', 'Yes'],
        'StreamingMovies': ['No internet service', 'Yes'],
        'Contract': ['One year', 'Two year'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Credit card (automatic)', 'Electronic check', 'Mailed check']
    }
    
    for feature, categories in categorical_features.items():
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    # Add binary feature
    feature_names.append('SeniorCitizen')
    
    # Get coefficients
    coefficients = model.named_steps['classifier'].coef_[0]
    
except FileNotFoundError:
    model = None
    feature_names = []
    coefficients = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    """Analytics dashboard with visualizations"""
    return render_template('analytics.html')

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64

@app.route('/feature_importance')
def feature_importance():
    """Return feature importance plot as base64"""
    if model is None or len(coefficients) == 0:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Create feature importance DataFrame
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False).head(15)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if coef > 0 else 'blue' for coef in feature_df['Coefficient']]
        
        bars = ax.barh(range(len(feature_df)), feature_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_df)))
        ax.set_yticklabels(feature_df['Feature'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Top 15 Feature Coefficients\n(Red = Increases Churn Risk, Blue = Decreases Churn Risk)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        img_base64 = create_plot_base64(fig)
        
        return jsonify({"plot": img_base64})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/model_performance')
def model_performance():
    """Return model performance visualization"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Create performance metrics visualization
        metrics = ['Recall', 'AUC-ROC', 'Accuracy', 'Precision']
        values = [79.7, 84.1, 74.2, 51.0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(metrics, values, color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'], alpha=0.8)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        # Confusion Matrix Simulation (for visualization)
        cm = np.array([[748, 287], [76, 298]])  # From your actual results
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Predicted: No Churn', 'Predicted: Churn'],
                   yticklabels=['Actual: No Churn', 'Actual: Churn'])
        ax2.set_title('Confusion Matrix')
        
        plt.tight_layout()
        img_base64 = create_plot_base64(fig)
        
        return jsonify({"plot": img_base64})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/business_insights')
def business_insights():
    """Return business insights visualization"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Risk factors and retention factors
        risk_factors = ['Month-to-month Contract', 'Fiber Optic Service', 'Electronic Check Payment', 
                       'High Monthly Charges', 'Low Tenure']
        risk_impact = [1.443, 0.622, 0.384, 0.158, 1.073]
        
        retention_factors = ['Two-year Contract', 'One-year Contract', 'Phone Service', 
                           'Online Security', 'Tech Support']
        retention_impact = [1.443, 0.778, 0.511, 0.463, 0.263]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Risk factors
        ax1.barh(risk_factors, risk_impact, color='red', alpha=0.7)
        ax1.set_xlabel('Impact Score (Coefficient Magnitude)')
        ax1.set_title('🔴 Top Churn Risk Factors')
        ax1.grid(axis='x', alpha=0.3)
        
        # Retention factors  
        ax2.barh(retention_factors, retention_impact, color='blue', alpha=0.7)
        ax2.set_xlabel('Impact Score (Coefficient Magnitude)')
        ax2.set_title('🔵 Top Customer Retention Factors')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        img_base64 = create_plot_base64(fig)
        
        return jsonify({"plot": img_base64})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        
        # Convert data to correct types and handle TotalCharges
        data['tenure'] = int(data['tenure'])
        data['MonthlyCharges'] = float(data['MonthlyCharges'])
        
        # Use TotalCharges from the frontend (now calculated dynamically)
        # or calculate it if not provided
        if 'TotalCharges' not in data or data['TotalCharges'] == '':
            data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
        else:
            data['TotalCharges'] = float(data['TotalCharges'])
            
        data['SeniorCitizen'] = int(data['SeniorCitizen'])
        
        # Ensure all required fields are present with defaults if missing
        required_fields = {
            'gender': 'Female',
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        for field, default_value in required_fields.items():
            if field not in data:
                data[field] = default_value
        
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

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        model_info = {
            "model_type": "Tuned Logistic Regression with SMOTE",
            "features": [
                "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"
            ],
            "performance": {
                "recall": "79.7%",
                "auc_roc": "84.1%",
                "accuracy": "74.2%",
                "precision": "51%"
            },
            "description": "Best model for identifying customers likely to churn"
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/bulk_predict', methods=['GET', 'POST'])
def bulk_predict():
    """Handle bulk predictions from CSV/Excel files"""
    if request.method == 'GET':
        return render_template('bulk_predict.html')
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload CSV or Excel files."}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the file based on extension
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:  # Excel files
                df = pd.read_excel(filepath)
        except Exception as e:
            os.remove(filepath)  # Clean up
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400
        
        # Validate required columns
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            os.remove(filepath)  # Clean up
            return jsonify({
                "error": f"Missing required columns: {missing_columns}",
                "required_columns": required_columns
            }), 400
        
        # Make predictions
        predictions = model.predict(df[required_columns])
        probabilities = model.predict_proba(df[required_columns])[:, 1]
        
        # Add predictions to dataframe
        df['Churn_Prediction'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]
        df['Churn_Probability'] = [f"{prob:.2%}" for prob in probabilities]
        df['Risk_Level'] = df['Churn_Probability'].apply(lambda x: 
            'High Risk' if float(x.strip('%')) >= 70 else 
            'Medium Risk' if float(x.strip('%')) >= 40 else 
            'Low Risk'
        )
        
        # Create output filename
        output_filename = f"predictions_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Save results
        if filename.lower().endswith('.csv'):
            df.to_csv(output_filepath, index=False)
        else:
            df.to_excel(output_filepath, index=False)
        
        # Generate summary statistics
        total_customers = len(df)
        churn_count = sum(predictions)
        churn_rate = (churn_count / total_customers) * 100
        
        risk_summary = df['Risk_Level'].value_counts().to_dict()
        
        # Clean up original file
        os.remove(filepath)
        
        return jsonify({
            "success": True,
            "summary": {
                "total_customers": total_customers,
                "predicted_churners": int(churn_count),
                "predicted_churn_rate": f"{churn_rate:.1f}%",
                "risk_distribution": risk_summary
            },
            "download_url": f"/download/{output_filename}",
            "message": f"Successfully processed {total_customers} customers"
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed results file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sample_template')
def download_sample_template():
    """Download a sample CSV template for bulk predictions"""
    try:
        # Create a sample template with required columns
        sample_data = {
            'gender': ['Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'No', 'Yes'],
            'tenure': [12, 24, 6],
            'PhoneService': ['Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'No', 'Yes'],
            'OnlineBackup': ['Yes', 'No', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'TechSupport': ['No', 'No', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No'],
            'StreamingMovies': ['No', 'Yes', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check'],
            'MonthlyCharges': [29.85, 56.95, 35.00],
            'TotalCharges': [358.20, 1367.80, 210.00]
        }
        
        df_sample = pd.DataFrame(sample_data)
        
        # Save sample template
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], 'churn_prediction_template.csv')
        df_sample.to_csv(template_path, index=False)
        
        return send_file(template_path, as_attachment=True, 
                        download_name='churn_prediction_template.csv')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Telco Churn Prediction API"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)