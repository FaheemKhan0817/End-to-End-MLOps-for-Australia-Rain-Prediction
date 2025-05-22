from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "artifacts/model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_rain_prediction_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

# Load the model and threshold
model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read())

# Define the 12 features
FEATURES = [
    'RainToday', 'Cloud3pm_missing', 'Sunshine', 'Humidity3pm',
    'Cloud3pm', 'Sunshine_missing', 'Cloud9am_missing', 'Rainfall',
    'WindGustDiff', 'Cloud9am', 'Pressure3pm', 'CloudCoverAvg'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {}
            for feature in FEATURES:
                value = request.form.get(feature)
                if feature in ['RainToday', 'Cloud3pm_missing', 'Sunshine_missing', 'Cloud9am_missing']:
                    # Handle categorical/binary features
                    input_data[feature] = int(value)
                else:
                    # Handle numerical features
                    input_data[feature] = float(value)

            # Convert input data to numpy array in the correct order
            input_array = np.array([[input_data[feature] for feature in FEATURES]])

            # Make prediction
            prob = model.predict_proba(input_array)[0, 1]
            probability = prob * 100  # Convert to percentage
            prediction = 1 if prob >= threshold else 0
            prediction_text = "Yes, it will rain tomorrow!" if prediction == 1 else "No, it won't rain tomorrow."

            return render_template('index.html', prediction=prediction_text, probability=probability)

        except Exception as e:
            error_message = f"Error: {str(e)}. Please check your inputs and try again."
            return render_template('index.html', prediction=error_message, probability=None)

    return render_template('index.html', prediction=None, probability=None)

if __name__ == '__main__':
    app.run(debug=True)