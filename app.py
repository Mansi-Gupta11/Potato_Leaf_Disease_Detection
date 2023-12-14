from flask import Flask, render_template, request, jsonify
import numpy as np

import cv2
from keras.models import load_model

app = Flask(__name__)




model = load_model('C:/Users/asus/Potato_Leaf_Disease_Detection/Streamlit/model/disease.h5')

CLASS_NAMES = ['Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight']


def process_image(file_bytes):
    opencv_image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), 1)
    opencv_image = cv2.resize(opencv_image, (256, 256))
    opencv_image = np.expand_dims(opencv_image, axis=0)
    return opencv_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            opencv_image = process_image(file.read())
            Y_pred = model.predict(opencv_image)
            result_index = np.argmax(Y_pred)
            result = CLASS_NAMES[result_index]
            return jsonify({'result': result})
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
