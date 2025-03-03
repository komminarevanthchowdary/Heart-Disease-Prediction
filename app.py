from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    data = list(data.values())
    data = [float(x) for x in data]

    # Convert to numpy array and reshape for prediction
    input_data = np.array(data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)
    result = "Positive" if prediction[0] == 1 else "Negative"

    return render_template("index.html", prediction_text=f"Heart Disease Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)