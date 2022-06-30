from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import pickle
import flask
from flask import Flask, jsonify, request
from PIL import Image
import io
from keras.models import load_model


model = pickle.load(open("mlp.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def upload_file():
  if request.method == 'POST':
    file_to_upload = request.files['file']
  
  img_bytes = file_to_upload.read()
  img = Image.open(io.BytesIO(img_bytes))
  img = img.resize((200, 200))
  img = img_to_array(img)
  img = img.reshape(1,200*200*3)
  a=model.predict(img)
  print(a)
  b=a.tolist()
  return jsonify(b)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')