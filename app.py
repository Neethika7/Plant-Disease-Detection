import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the model
model = load_model('model.h5')
print('Model loaded successfully.Check http://127.0.0.1:5000/')

# Label mapping
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image):
    img = image.resize((225, 225))  # Ensure the image is resized to the expected input size
    x = img_to_array(img)
    x = x.astype('float32') / 255.  # Normalizing the input
    x = np.expand_dims(x, axis=0)  # Expanding dimensions to match model input shape
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        if file:
            # Read the image directly from the file stream in memory
            image = Image.open(BytesIO(file.read()))
            predictions = getResult(image)
            predicted_label = labels[np.argmax(predictions)]
            return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
