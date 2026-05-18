from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("best_house_price_model.pkl")

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    overallqual = float(request.form["overallqual"])

    grlivarea = float(request.form["grlivarea"])

    garagecars = float(request.form["garagecars"])

    totalbsmtsf = float(request.form["totalbsmtsf"])

    fullbath = float(request.form["fullbath"])

    yearbuilt = float(request.form["yearbuilt"])


    features = np.array([[
        overallqual,
        grlivarea,
        garagecars,
        totalbsmtsf,
        fullbath,
        yearbuilt
    ]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ${prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)