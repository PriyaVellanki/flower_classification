import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
import urllib.request
from io import BytesIO
from PIL import Image


classes = ['daisy',
           'dandelion']



def preprocess_input(image):
    # img = Image.open(image)
    img = image.resize((160,160),Image.NEAREST)

    x = np.array(img,dtype='float32')

    X = np.array([x])
    X /= 127.5
    X -= 1.0

    return X


   

def flower_classification(image):
  X = preprocess_input(image)
  
  interpreter = tflite.Interpreter(model_path='./models/flower_classification.tflite')
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]['index']
  
  output_index = interpreter.get_output_details()[0]['index']
  

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()
  pred_lite = interpreter.get_tensor(output_index)
  pred = pred_lite[0].tolist()
  
  return dict(zip(classes, pred))

def main():
   
   st.title("Daisy or Dandelion Flower Classification")
   image_file = st.file_uploader("Upload Image for classification", type=['jpg', 'jpeg'])
   if image_file is not None:
        input_image = Image.open(image_file)
        st.text("Uploading and processing image")
        st.image(input_image)

   if st.button("Predict"):
        result= flower_classification(input_image)
        st.write(result)


if __name__ == '__main__':
    main()

