'''Main'''
import re

import face_recognition
import numpy as np
import pandas as pd
import os

from PIL import Image
from sklearn.linear_model import LogisticRegression

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import matplotlib as mpl

import csv
from collections import Counter

'''Data Prep'''
from sklearn import preprocessing as pp
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

'''Algos'''
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb


csvfile = "primemod.csv"
im_cam = 'img/test-img-3.jpg'

#FROM USERS NOMILIZED FACE- CREATE DATA ARRAY
image = face_recognition.load_image_file(im_cam)
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    res = (face_landmarks['bottom_lip'] + face_landmarks[
        'top_lip'])  # change the 0 for anything but smiling, and 1 for smiling
    resx = ", ".join(repr(e) for e in res)
    read = str(resx).replace('(', '').replace(')', '')
    print(read)

    # with open(csvfile, "a") as output:
    #     writer = csv.writer(output)
    #     writer.writerows([res])
