import face_recognition
import numpy as np
from PIL import Image

# class halseyCamera:


### CHANGE THE DIR TO MATCH THE CAMERAS SETUP ###
file = 'img/test-img-1.jpg'  # ONLY FOR TESTING

## LEAVE THE IMPORT AS A GLOBAL  FUNCTION ~~~~~~~###
im = face_recognition.load_image_file(file)
face_cords = face_recognition.face_locations(im)
### ^^ LOAD THE IMAGE return CROP COORDINATES ^^ ###


## CREATE LOOP INDEX FROM NUMBER OF PAX IN PHOTO
def face_count(cords):
    numb = len(cords)
    count = numb
    return count


def convert(string):
    li = list(string.split("-"))
    return li


### NORMALIZE THE FACES
def normalize(count, cords, image):
    top, right, bottom, left = cords[count]
    face_array = image[top:bottom, left:right]
    face_conv = Image.fromarray(face_array).convert('L').resize((64, 64))
    x = np.array(face_conv)
    return x


# FROM USERS NORMALIZED FACE /  CREATE DATA LIST for COMPARISON
def camera_data(norm):
    face_landmarks_list = face_recognition.face_landmarks(norm)
    pil_image = Image.fromarray(norm)
    for face_landmarks in face_landmarks_list:
        res = (face_landmarks['bottom_lip'] + face_landmarks[
            'top_lip'])
        resx = ", ".join(repr(e) for e in res)
        read = str(resx).replace('(', '').replace(')', '')
        readx = convert(read)
        return readx


vara = face_count(face_cords)
# print("Index, Faces:  {}".format(vara))
# print("Face Cordinates:  {}".format(face_cords))

for index in range(vara):
    varb = normalize(index, face_cords, im)
    varc = camera_data(varb)
    print("Vectors:  {}".format(varc))

# print(type(varc))
