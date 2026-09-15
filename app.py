from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model, scaler, and PCA
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":

        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        pedigree = float(request.form["pedigree"])
        age = float(request.form["age"])

        # Arrange input in the same order as training
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]])

        # Apply scaler
        input_scaled = scaler.transform(input_data)

        # Apply PCA
        input_pca = pca.transform(input_scaled)

        # Prediction
        prediction_result = model.predict(input_pca)[0]
        probability = model.predict_proba(input_pca)[0][1]*100

        if prediction_result == 1:
            result = "High Risk of Diabetes"
        else:
            result = "Low Risk of Diabetes"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 2)
        )

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)
