from PIL import Image, ImageDraw
import face_recognition
# Load the jpg file into a numpy array

src = "/home/admin/flitwick/halsey/img/test-img-3.jpg"
image = face_recognition.load_image_file(src)
# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)

for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=1)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=1)

    # Gloss the lips
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=1)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=1)

    pil_image.show()
