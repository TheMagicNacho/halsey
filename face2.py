from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
src = "/home/admin/flitwick/halsey/img/test-img-1.jpg"
#pre_load = load_img(src, grayscale=True)
image = face_recognition.load_image_file(src)
#im = np.array(Image.open('data/src/lena_square.png').convert('L').resize((256, 256)))
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
face_array = image[top:bottom, left:right]
face_conv = Image.fromarray(face_array).convert('L').resize((48, 48))
#face_image = np.array(face_conv)

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}


model = load_model("models/emotions.hdf5")
predicted_class = np.argmax(model.predict(face_conv))

label_map = dict((v,k) for k,v in emotion_dict.items())
predicted_label = label_map[predicted_class]

print(predicted_label)