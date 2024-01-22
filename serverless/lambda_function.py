import json
import numpy as np
import tflite_runtime.interpreter as tflite
from urllib import request
from io import BytesIO
from PIL import Image



classes = ['daisy',
           'dandelion']


def lambda_handler(event, context):
  url = event['url']
  result = prediction(url)

  return result


def preprocess_input(x):
  x /= 127.5
  x -= 1.0
  return x


def download_image(url):
  with request.urlopen(url) as res:
    buffer = res.read()
  data = BytesIO(buffer)
  img = Image.open(data)
  img = img.resize((160,160),Image.NEAREST)
  return img

def convert_img_np(url):
  img = download_image(url)
  x = np.array(img,dtype='float32')
  X = np.array([x])
  X = preprocess_input(X)

  return X


def prediction(url):
  X = convert_img_np(url)
  
  interpreter = tflite.Interpreter(model_path='flower_classification.tflite')
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]['index']
  
  output_index = interpreter.get_output_details()[0]['index']
  

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()
  pred_lite = interpreter.get_tensor(output_index)
  pred = pred_lite[0].tolist()
  
  return dict(zip(classes, pred))
  











