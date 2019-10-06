from imutils import face_utils
from scipy.spatial import distance as dist
#from imutils import face_utils
#import numpy as np
import time
import dlib
from PIL import Image
import face_recognition


def smile(mouth):
    a = dist.euclidean(mouth[3], mouth[9])
    b = dist.euclidean(mouth[2], mouth[10])
    c = dist.euclidean(mouth[4], mouth[8])
    avg = (a + b + c) / 3
    d = dist.euclidean(mouth[0], mouth[6])
    mar = avg / d
    return mar


COUNTER = 0
TOTAL = 0

shape_predictor = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while True:
    src = "/home/admin/flitwick/halsey/img/test-img-1.jpg"
    # pre_load = load_img(src, grayscale=True)
    image = face_recognition.load_image_file(src)
    # im = np.array(Image.open('data/src/lena_square.png').convert('L').resize((256, 256)))
    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    face_array = image[top:bottom, left:right]
    face_conv = Image.fromarray(face_array).convert('L').resize((48, 48))
    rects = detector(face_conv, 0)
    for rect in rects:
        shape = predictor(face_conv, rect)
        shape = face_utils.FACIAL_LANDMARKS_IDXS["mouth"].shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = smile(mouth)
        # mouthHull = cv2.convexHull(mouth)
        # print(shape)
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mar <= .3 or mar > .38:
            COUNTER += 1
        else:
            if COUNTER >= 15:
                TOTAL += 1
                # frame = vs.read()
                time.sleep(.3)
                # frame2 = frame.copy()
                img_name = "opencv_frame_{}.png".format(TOTAL)
                # cv2.imwrite(img_name, frame)
                print("smiled")
            COUNTER = 0

    # cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
