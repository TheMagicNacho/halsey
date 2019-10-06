from PIL import Image
from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
import face_recognition


src = "/home/admin/flitwick/halsey/img/test-img-1.jpg"

#pre_load = load_img(src, grayscale=True)
image = face_recognition.load_image_file(src)
x = face_recognition.face_locations(image)

s = len(x)
print(s)




# top, right, bottom, left = face_locations[2]
# face_array = image[top:bottom, left:right]
# face_conv = Image.fromarray(face_array).convert('L').resize((64, 64))
# x = np.array(face_conv)
# plt.imshow(x)
# plt.show()
#
# model = load_model("xxx")