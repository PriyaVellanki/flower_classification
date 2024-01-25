import urllib.request
from io import BytesIO

import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request
from PIL import Image

classes = ["daisy", "dandelion"]


app = Flask("flower-classification")


def preprocess_input(url):
    with urllib.request.urlopen(url) as res:
        buffer = res.read()
    data = BytesIO(buffer)
    img = Image.open(data)
    img = img.resize((160, 160), Image.NEAREST)

    x = np.array(img, dtype="float32")

    X = np.array([x])
    X /= 127.5
    X -= 1.0

    return X


def flower_classification(url):
    X = preprocess_input(url)

    interpreter = tflite.Interpreter(
        model_path="../models/flower_classification.tflite"
    )
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]

    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred_lite = interpreter.get_tensor(output_index)
    pred = pred_lite[0].tolist()

    return dict(zip(classes, pred))


@app.route("/flower-classification", methods=["POST"])
def predict():
    data = request.get_json()
    url = data["url"]
    result = flower_classification(url)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
