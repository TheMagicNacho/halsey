'''Main'''
import numpy as np
import pandas as pd
import os

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


# Acquire Data
current_path = os.getcwd()
file = 'models/halseytrainer.csv'
data = pd.read_csv(file)

# # print(data.columns)
dataX = data.copy().drop(['class'], axis=1)
dataY = data['class'].copy()

featuresToScale = dataX.drop(['file'], axis=1).columns

sX = pp.StandardScaler(copy=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])
scalingFactors = pd.DataFrame(data=[sX.mean_, sX.scale_], index=['Mean', 'StDev'], columns=featuresToScale)

correlationMatrix = pd.DataFrame(data=[], index=dataX.columns, columns=dataX.columns)


k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, random_state=random_state, solver=solver, n_jobs=n_jobs)

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[], index=y_train.index, columns=[0, 1])

model = logReg

print(model)