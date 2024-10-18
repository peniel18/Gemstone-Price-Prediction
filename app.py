from flask import Flask, render_template, jsonify , request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict_datapoint', methods = ["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else: 
        data = CustomData(
            carat = float(request.form.get("carat")), 
            depth = float(request.form.get("depth")),
            table = float(request.form.get("table")), 
            x = float(request.form.get("X")),
            y = float(request.form.get("Y")),
            z = float(request.form.get("Z")),
            color = request.form.get("color"),
            clarity = request.form.get("clarity"),     
        )


if __name__ == "__main__":
    app.run(debug=True)

