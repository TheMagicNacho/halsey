"""
This script will find all the faces in a photo then crop them down.
The croped down photo is noramlized to match the data set.
"""

from PIL import Image
from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
import face_recognition




src = "/home/admin/flitwick/halsey/img/test-img-2.jpg"

#pre_load = load_img(src, grayscale=True)
image = face_recognition.load_image_file(src)

#im = np.array(Image.open('data/src/lena_square.png').convert('L').resize((256, 256)))

face_locations = face_recognition.face_locations(image)


top, right, bottom, left = face_locations[0]
face_array = image[top:bottom, left:right]
face_conv = Image.fromarray(face_array).convert('L').resize((48, 48))
x = np.array(face_conv)



plt.imshow(x)
plt.show()

model = load_model("xxx")


