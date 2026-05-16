# =============================================================
# CREDIT CARD FRAUD DETECTION - FLASK WEB APPLICATION
# =============================================================
# This script creates a web server where users can enter
# transaction details and get a prediction: Fraud or Genuine
# =============================================================

# ---- Import required libraries ----
from flask import Flask, render_template, request
import joblib       # For loading the saved ML model
import numpy as np  # For creating the input array
import os           # For checking file paths

# ---- Create the Flask application ----
app = Flask(__name__)

# -----------------------------------------------
# LOAD THE TRAINED MODEL (runs once at startup)
# -----------------------------------------------
MODEL_PATH   = "models/fraud_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

# Check if model exists (user must train first)
if not os.path.exists(MODEL_PATH):
    print("\n❌ ERROR: Trained model not found!")
    print("Please run:  python train_model.py  first!")
    exit()

# Load the model from disk
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print(f"✅ Model loaded! Features expected: {len(feature_names)}")


# -----------------------------------------------
# ROUTE 1: HOME PAGE (shows the input form)
# -----------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """
    Shows the main page with the transaction input form.
    GET = user visits the page → show the form.
    """
    return render_template("index.html", prediction=None)


# -----------------------------------------------
# ROUTE 2: PREDICT (processes the form submission)
# -----------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives the transaction data from the form,
    runs the ML model, and shows the result.
    POST = user submitted the form → predict and show result.
    """
    try:
        # ---- Collect all input values from the HTML form ----
        # The credit card dataset has 30 features:
        # Time, V1-V28 (PCA features), Amount

        # Get Time (seconds since first transaction)
        time_val = float(request.form.get("Time", 0))

        # Get V1 to V28 (these are anonymized PCA features)
        v_values = []
        for i in range(1, 29):
            v = float(request.form.get(f"V{i}", 0))
            v_values.append(v)

        # Get Amount (transaction amount in dollars)
        amount_val = float(request.form.get("Amount", 0))

        # ---- Combine into one list in the correct order ----
        # Order must match training data: Time, V1-V28, Amount
        input_data = [time_val] + v_values + [amount_val]

        # Convert to numpy array (required by sklearn)
        input_array = np.array(input_data).reshape(1, -1)

        # ---- Make Prediction ----
        prediction = model.predict(input_array)[0]

        # Get the probability (confidence score)
        probability = model.predict_proba(input_array)[0]
        confidence = max(probability) * 100  # Convert to percentage

        # ---- Prepare result message ----
        if prediction == 1:
            result = "FRAUDULENT"
            result_class = "fraud"       # CSS class for red styling
            icon = "⚠️"
            message = "This transaction appears to be FRAUDULENT. Please review immediately."
        else:
            result = "GENUINE"
            result_class = "genuine"     # CSS class for green styling
            icon = "✅"
            message = "This transaction appears to be GENUINE. No action required."

        # ---- Render the page with prediction result ----
        return render_template(
            "index.html",
            prediction=result,
            result_class=result_class,
            icon=icon,
            message=message,
            confidence=f"{confidence:.2f}",
            amount=amount_val,
            time_val=time_val
        )

    except Exception as e:
        # If something goes wrong, show an error message
        return render_template(
            "index.html",
            prediction="ERROR",
            result_class="error",
            icon="❌",
            message=f"Something went wrong: {str(e)}. Please check your inputs.",
            confidence="N/A",
            amount=0,
            time_val=0
        )


# -----------------------------------------------
# MAIN ENTRY POINT - Run the Flask App
# -----------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🚀 Credit Card Fraud Detection App Running!")
    print("  Open your browser and go to:")
    print("  👉 http://127.0.0.1:5000")
    print("=" * 50 + "\n")

    # debug=True means: auto-reload on code changes (useful during development)
    app.run(debug=True)
