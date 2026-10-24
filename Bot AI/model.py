from ultralytics import YOLO

detector = YOLO("yolo11n.pt")


def hay_un_gato(image_path):
    resultados = detector(image_path, verbose=False)

    for resultado in resultados:
        for objeto in resultado.boxes:
            clase = int(objeto.cls[0])
            confianza = float(objeto.conf[0])

            if clase == 15 and confianza >= 0.40:
                return True

    return False


import tf_keras as keras
from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def get_class(model_path, labels_path, image_path):
    np.set_printoptions(suppress=True)

    model = load_model(model_path, compile=False)

    with open(labels_path, "r", encoding="utf-8") as file:
        class_names = file.readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(
        image, (224, 224), Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)
    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data[0] = normalized_image_array
    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction[0])
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name[2:].strip(), confidence_score
