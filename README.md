# Telco Customer Churn Prediction

This project develops a machine learning model to predict customer churn for a telecommunications company. It includes a full data analysis pipeline in a Jupyter Notebook and a web application built with Flask that uses the trained model to make real-time predictions.


*(Note: You should replace this with a screenshot of your own running application!)*

## Project Objective

Customer churn is a major concern for telecommunications companies, as acquiring new customers is significantly more expensive than retaining existing ones. The goal of this project is to build a highly accurate model that predicts which customers are likely to churn. This allows the business to take proactive measures, such as offering targeted promotions, to reduce revenue loss and improve customer loyalty.

## Features

-   **Detailed Exploratory Data Analysis (EDA):** The `Project_Report.ipynb` notebook contains a comprehensive analysis of the customer data, uncovering key factors that influence churn.
-   **Machine Learning Model Pipeline:** A robust `scikit-learn` pipeline was built to handle data preprocessing (scaling, encoding), class imbalance (using SMOTE), and model training.
-   **Interactive Web Application:** A web interface built with **Flask** allows users to input customer data and receive an instant churn prediction, demonstrating the model's real-world applicability.

## Project Structure

```
.
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── Project_Report.ipynb
├── Telco-Customer-Churn.csv
├── app.py
├── churn_model.joblib
├── requirements.txt
└── README.md
```

## Technology Stack

-   **Backend:** Python, Flask
-   **Frontend:** HTML, CSS
-   **Data Science & ML:** Pandas, NumPy, Scikit-learn, Imbalanced-learn, Joblib
-   **Data Visualization:** Matplotlib, Seaborn

## Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/T-srikrishna/telco-churn-prediction.git
    cd telco-churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Run the Web Application

The repository includes a pre-trained model (`churn_model.joblib`), so you can run the web application directly.

From the project's root directory, run the Flask app:

```bash
flask run
```
or
```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000`. You can now input customer details into the form and click "Predict Churn" to get a result.

### 2. Explore the Analysis & Model Training

To see the full data analysis, model training, and evaluation process, you can run the Jupyter Notebook:
```bash
jupyter notebook "Project_Report.ipynb"
```

## Final Model Details

The champion model selected for this project is a **Tuned Logistic Regression** model. It was chosen for three key reasons:

1.  **Highest Recall (79.7%):** It was the best model at identifying customers who will actually churn. For a retention campaign, it is far more costly to miss a potential churner (a false negative) than it is to mistakenly target a loyal customer (a false positive).
2.  **High Interpretability:** Its linear nature makes it easy to understand and explain which factors (e.g., contract type, monthly charges) are influencing the predictions.
3.  **Simplicity and Speed:** It is computationally efficient, making it fast to train and deploy.

The model was trained using a full pipeline incorporating data preprocessing (StandardScaler for numerical features, OneHotEncoder for categorical features) and SMOTE (Synthetic Minority Over-sampling Technique) to effectively handle the class imbalance in the dataset.
