🍔 Food Classification Using Deep Learning

A deep learning-based web application that classifies food images into different categories and displays nutritional information such as calories, protein, carbohydrates, and fat.

Users can upload an image of a food item, and the application predicts the food category (for example, Pizza, Burger, Biryani, and Dosa) with a confidence score.

📌 Features
📷 Upload food images (JPG, JPEG, PNG)
🤖 Predict food category using a trained deep learning model
📊 Display prediction confidence
🥗 Show nutritional information
🌐 Web-based user interface
☁️ Ready for future deployment

🧠 Technologies Used
Python
TensorFlow
Keras
MobileNetV2
HTML
CSS
Flask
NumPy
Pandas
Pillow


The project uses the Food-101 dataset and can be extended with custom food classes.

Example classes:

Pizza
Burger
Biryani
Dosa
Idli
Samosa
Pasta


🏗️ Model Architecture

The application uses transfer learning with MobileNetV2.

Training Workflow
Load and preprocess food images
Resize images to 224 × 224 pixels
Apply data augmentation
Use MobileNetV2 as the base model
Add custom dense layers
Train the model
Save the trained model as food_classifier.h5

📊 Example Prediction
Attribute	Value
Predicted Class	Biryani
Confidence	96.8%
Calories	320 kcal
Protein	12 g
Carbohydrates	42 g
Fat	10 g

⚙️ Installation
1. Clone the Repository
git clone https://github.com/Sanjanadamarla/food-classification-app.git
cd food-classification-app
2. Create a Virtual Environment
python -m venv venv
3. Activate the Virtual Environment

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt
5. Run the Application
python app.py
6. Open in Browser
http://127.0.0.1:5000/

📋 Requirements
Flask
tensorflow
keras
numpy
pandas
pillow
scikit-learn
matplotlib

🎯 Future Enhancements
Multi-food detection in a single image
Calorie tracking and meal recommendations
Mobile application development
Cloud deployment
