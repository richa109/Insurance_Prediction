from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define the expected feature columns as used during training
feature_cols = [
    "Age",
    "Is_Senior",
    "Married_Premium_Discount",
    "Prior_Insurance",
    "Prior_Insurance_Premium_Adjustment",
    "Claims_Frequency",
    "Claims_Severity",
    "Claims_Adjustment",
    "Policy_Adjustment",
    "Premium_Amount",
    "Safe_Driver_Discount",
    "Multi_Policy_Discount",
    "Bundling_Discount",
    "Total_Discounts",
    "Time_Since_First_Contact",
    "Website_Visits",
    "Inquiries",
    "Quotes_Requested",
    "Time_to_Conversion",
    "Credit_Score",
    "Premium_Adjustment_Credit",
    "Premium_Adjustment_Region",
    "Marital_Status_Divorced",
    "Marital_Status_Married",
    "Marital_Status_Single",
    "Marital_Status_Widowed",
    "Policy_Type_Full Coverage",
    "Policy_Type_Liability-Only",
    "Source_of_Lead_Agent",
    "Source_of_Lead_Online",
    "Source_of_Lead_Referral",
    "Region_Rural",
    "Region_Suburban",
    "Region_Urban"
]

@app.route("/")
def index():
    return render_template("index.html", feature_cols=feature_cols)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Build a dictionary from submitted form data
        input_data = {}
        for col in feature_cols:
            # Get each value from the form. Ensure no value is missed.
            value = request.form.get(col)
            if value is None or value == "":
                return f"Error: missing value for {col}"
            input_data[col] = float(value)
        
        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Make predictions using the loaded model
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        return render_template("result.html", prediction=prediction, proba=prediction_proba)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
