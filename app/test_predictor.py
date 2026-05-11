import os
from predictor import predict_food

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exact image path
image_path = os.path.join(
    BASE_DIR,
    "dataset",
    "food11",
    "evaluation",
    "Rice",
    "7.jpg"
)

print("Testing image:", image_path)

# Predict
result = predict_food(image_path)

print("\nPredicted Food:", result["food_name"])
print("Confidence:", result["confidence"], "%")
print("Nutrition Info:")
print(result["nutrition"])