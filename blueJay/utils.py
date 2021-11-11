import os
import base64
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGES_PATH_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/images/"
MODEL_PATH_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model/"
SPECIES_PATH_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/raw_data/"

MODEL_PATH_remote = "gs://liquid-projects/model/"

# #############################################################################
# #########################   get_image_decode   ##############################
# #############################################################################
def get_image_decode(image_name):
    with open(os.path.join(IMAGES_PATH_local, image_name), "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
        s = my_string.decode('UTF-8')
    return s

# #############################################################################
# #########################   predict_specie     ##############################
# #############################################################################
def predict_specie(requested_image):

    # loading model
    model = load_model(MODEL_PATH_local)

    # loading classes indices
    with open(os.path.join(SPECIES_PATH_local, "classes_indices.json")) as json_file:
       classes_indices = json.load(json_file)

    # image preprocessing
    image_to_predict = np.array(requested_image.resize((224, 224))).reshape(-1, 224, 224, 3)
    image_preprocessed = preprocess_input(image_to_predict)

    # prediction
    prediction = model.predict(image_preprocessed)
    predicted_val = np.argmax(prediction)
    predicted_str = list(classes_indices.keys())[list(classes_indices.values()).index(predicted_val)]
    return [predicted_str, "\n\nAccuracy: ", str(np.max(prediction)*100)[0:5] + "%"]
