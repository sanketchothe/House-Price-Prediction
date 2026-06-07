from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
# Load saved files
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        input_data = {
            "longitude": float(data["longitude"]),
            "latitude": float(data["latitude"]),
            "housing_median_age": float(data["housing_median_age"]),
            "total_rooms": float(data["total_rooms"]),
            "total_bedrooms": float(data["total_bedrooms"]),
            "population": float(data["population"]),
            "households": float(data["households"]),
            "median_income": float(data["median_income"]),
            "ocean_proximity": data["ocean_proximity"]
        }
        df = pd.DataFrame([input_data])

        transformed = pipeline.transform(df)
        prediction = model.predict(transformed)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ${round(prediction, 2)}"
        )
    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == "__main__":
    app.run(debug=True)