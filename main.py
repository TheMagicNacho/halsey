import csv
import face_recognition
import numpy as np
from PIL import Image

### CHANGE THE DIR TO MATCH THE CAMERAS SETUP ###
file = 'img/test-img-3.jpg'  # ONLY FOR TESTING

## LEAVE THE IMPORT AS A GLOBAL  FUNCTION ~~~~~~~###
std_frac = 3  # fraction of the standard deviation. The higher the number the more strict the algorithim
im = face_recognition.load_image_file(file)
face_cords = face_recognition.face_locations(im)


### ^^ LOAD THE IMAGE return CROP COORDINATES ^^ ###


## CREATE LOOP INDEX FROM NUMBER OF PAX IN PHOTO
def face_count(cords):
    numb = len(cords)
    count = numb - 1
    return count


def convert(string):
    li = list(string.split('-'))
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
        resx = ', '.join(repr(e) for e in res)
        read = str(resx).replace(',', '').replace('(', '').replace(')', '')
        readx = read.split()
        return readx

#COMPARE THE VECTORS BETWEEN TRAINNING MODEL AND INPUT IMAGE RETURN A SCORE
def compare(cam_array):
    file = 'models/happyvectors.csv'
    with open(file, 'r') as model:
        mr = csv.reader(model)
        modelrow = list(mr)
    mean = modelrow[2]
    std = modelrow[3]
    # print(type(std))
    for i in range(48):  # Changed to 2 for debugging, needs to be 48
        local_array = int(float(cam_array[i]))
        model_array = int(float(mean[i]))
        std_array = int(float(std[i])) / std_frac
        lower_limit = model_array - std_array
        upper_limit = model_array + std_array
        if local_array < upper_limit or local_array < lower_limit:
            # print('SMILE DETECTED')
            return int(1)
        else:
            # print('No Smile :(')
            return int(0)


index = face_count(face_cords)
norm_im = normalize(index, face_cords, im)
cam_to = camera_data(norm_im)
compare(cam_to)
