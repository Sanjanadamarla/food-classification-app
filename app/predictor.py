import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from labels import CLASS_NAMES

# ==========================
# Paths
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "food_classifier.keras")
NUTRITION_PATH = os.path.join(BASE_DIR, "nutrition.csv")

# ==========================
# Load Model and Nutrition Data
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)
nutrition_df = pd.read_csv(NUTRITION_PATH)

# ==========================
# Image Preprocessing
# ==========================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array

# ==========================
# Nutrition Lookup
# ==========================
def get_nutrition(food_name):
    row = nutrition_df[nutrition_df["food_name"] == food_name]

    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "calories": row["calories"],
        "protein": row["protein"],
        "carbs": row["carbs"],
        "fat": row["fat"],
        "health_tip": row["health_tip"]
    }

# ==========================
# Main Prediction Function
# ==========================
def predict_food(image_path):
    image = preprocess_image(image_path)

    predictions = model.predict(image, verbose=0)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    food_name = CLASS_NAMES[class_index]
    nutrition = get_nutrition(food_name)

    return {
        "food_name": food_name,
        "confidence": round(confidence, 2),
        "nutrition": nutrition
    }