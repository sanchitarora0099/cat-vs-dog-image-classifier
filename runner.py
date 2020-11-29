import cv2
import tensorflow as tf
import os
from os import listdir

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 50 # same as the training data set
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("cat-vs-dog.model")

predict_dir_path='testlite/'
onlyfiles = [f for f in listdir(predict_dir_path) if os.path.isfile(os.path.join(predict_dir_path, f))]

 
for file in onlyfiles:
    prediction = model.predict([prepare(predict_dir_path+file)])
    print(file + ':' + CATEGORIES[int(prediction[0][0])])
