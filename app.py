from flask import Flask, request, jsonify
import io, os, base64
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras.applications import VGG19
from tensorflow.keras import models as tf_models, layers
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

MODEL_ID = "1hWurEL1EMB_adJOovXG5LVcZUFn9l_wS"
MODEL_PATH = "best_weights_fold_5.weights.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

# Build model
input_shape = (224, 224, 3)
base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base.layers[:15]: layer.trainable = False

model = tf_models.Sequential([
    layers.Input(shape=input_shape),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Reshape((1, -1)),
    layers.LSTM(128),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(4, activation='softmax')
])
model.load_weights(MODEL_PATH)

def preprocess(img):
    img = img.resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert('RGB')
        processed = preprocess(img)
        pred = model.predict(processed)
        pred_class = int(np.argmax(pred, axis=1).item())
        return jsonify({'predicted_class': pred_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
