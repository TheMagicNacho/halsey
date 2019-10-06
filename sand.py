'''
Create vector CSV for all smiles within a directory
ONLY USE TO CREATE A DATABASE FOR SUPERVISED TRAINING
'''
from PIL import Image
import face_recognition
import csv
import glob

csvfile = "primemod.csv"


for file in glob.glob('models/SMILEsmileD-master/SMILEs/positives/positives7/*.jpg'):
    image = face_recognition.load_image_file(file)
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        res = (face_landmarks['bottom_lip'] + face_landmarks['top_lip'])
        with open(csvfile, "a") as output:
            writer = csv.writer(output)
            writer.writerows([res])
