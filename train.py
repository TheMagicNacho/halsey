"""
ANALYSIS FOR FER2013

### OUTPUTS
    FER2013 ANALYSIS
            emotion
count  35887.000000
mean       3.323265
std        1.873819
min        0.000000
25%        2.000000
50%        3.000000
75%        5.000000
max        6.000000

###COLUMNS
Index(['emotion', 'pixels', 'Usage'], dtype='object')

###HEADDER
   emotion                                             pixels     Usage
0        0  70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...  Training
1        0  151 150 147 155 148 133 111 140 170 174 182 15...  Training
2        2  231 212 156 164 174 138 161 173 182 200 106 38...  Training
3        4  24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...  Training
4        6  4 0 0 0 0 0 0 0 0 0 0 0 3 15 23 28 48 50 58 84...  Training


## FREQ OF USE
3    8989 (happy)
6    6198 (neutral)
4    6077 (sad)
2    5121
0    4953
5    4002
1     547

"""

'''Main'''
import numpy as np
import pandas as pd
import os

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

# HERE LIES THE DICTIONARY TYPEScategories(0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral).

# Acquire Data
current_path = os.getcwd()
file = 'models/fer2013.csv'
data = pd.read_csv(file)
# c = data['emotion'].value_counts()

dataX = data.copy().drop(['emotion'], axis=1)
dataY = data['emotion'].copy()

featuresToScale = dataY.drop(['Usage'], axis=1).columns
sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])
scalingFactors = pd.DataFrame(data=[sX.mean_, sX.scale_], index=['Mean', 'StDev'], columns=featuresToScale)

print(scalingFactors)
# print(dataY)
