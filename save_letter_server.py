from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__, static_folder='web')  # Assume all web files are in the 'web' directory

# Correct path for model loading
model = tf.keras.models.load_model(r'D:\CapstoneProject\Capstone_Code\619.h5')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_file(path):
    return send_from_directory(app.static_folder, path)


@app.route('/predict-letter', methods=['POST'])
def predict_letter():
    file = request.files['userLetter']
    if file:
        image = Image.open(io.BytesIO(file.read())).convert('L')
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 64, 64, 1)  # Add batch dimension

        try:
            prediction = model.predict(image_array)
            print(f'Prediction raw output: {prediction}')
            predicted_class = np.argmax(prediction, axis=1)
            print(f'Predicted class index: {predicted_class}')
            confidence = np.max(prediction, axis=1)
            print(f'Confidence score: {confidence}')

            # Assuming you have class names mapped to class indices
            class_names = ['A', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң',
                           'О', 'Ө', 'П', 'Р', 'С', 'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
                           'І', 'Ь', 'Э', 'Ю', 'Я']

            if len(predicted_class) > 0:
                predicted_letter = class_names[predicted_class[0]]
                confidence = float(confidence[0])

                return jsonify({
                    'status': 'success',
                    'predicted_letter': predicted_letter,
                    'confidence': f"{confidence * 100:.2f}%"
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Model did not return any predictions'
                }), 500
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    else:
        return {'status': 'error', 'message': 'No image received'}, 400


if __name__ == '__main__':
    app.run(debug=True)
