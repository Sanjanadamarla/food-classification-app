import os
from flask import Flask, render_template, request
from predictor import predict_food

# ==========================
# Flask App Configuration
# ==========================
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================
# Home Route
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Prediction Route
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image.")

    # Save uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Run prediction
    result = predict_food(file_path)

    # Relative path for displaying image in browser
    image_url = f"uploads/{file.filename}"

    return render_template(
        "index.html",
        result=result,
        image_url=image_url
    )


# ==========================
# Run Application
# ==========================
if __name__ == "__main__":
    app.run(debug=True)