from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename



model = load_model('C:/Users/asus/Potato_Leaf_Disease_Detection/Streamlit/model/disease.h5')  # Replace with the actual path
disease_names = {0: 'Potato___Early_blight', 1: 'Potato___healthy', 2: 'Potato___Late_blight'}

def process_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML template for your homepage

@app.route('/predict', methods=['POST'])
def predict():
    try:

        if request.method == 'POST':
            file = request.files['file']
            if file:
                img_path = 'C:/Users/asus/Potato_Leaf_Disease_Detection/uploads/uploaded_imagr.jpg'  # Replace with the actual path
                file.save(img_path)
                processed_img = process_image(img_path)
                prediction = model.predict(processed_img)
                print(f"Raw Predictions: {prediction}")
                result = np.argmax(prediction)
                return jsonify({'result': result.item()})
    except Exception as e:
            # Log the exception details
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

