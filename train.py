"""
ANALYSIS FOR Smile Master Corpus. I can't remember where I got the database from.
Images at 64 x 64

### OUTPUTS
(6599, 25) = MODEL SHAPE
6599 individual files studied
25 pairs of vectors

1842 Happy faces
##### HAPPY ONLY VECTORS
,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,class,file
count,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0,1842.0
mean,46.77524429967427,47.45819761129207,41.587404994571116,53.127035830618894,35.756786102062975,55.32736156351792,31.184039087947884,55.639522258414765,26.719326818675352,55.087404994571116,21.13300760043431,52.57111834961998,16.42725298588491,46.57654723127036,18.32356134636265,46.9771986970684,27.01628664495114,50.754614549402824,31.305646036916396,51.29153094462541,35.732899022801305,50.98642779587405,44.89033659066232,47.762214983713356,16.42725298588491,46.57654723127036,21.733984799131377,44.646036916395225,27.40445168295331,44.011943539630835,31.47502714440825,45.05157437567861,35.72475570032573,44.24972855591748,41.452225841476654,45.244299674267104,46.77524429967427,47.45819761129207,44.89033659066232,47.762214983713356,35.71226927252986,46.91205211726384,31.403365906623236,47.22529858849077,27.252442996742673,46.64549402823018,18.32356134636265,46.9771986970684,1.0,921.5
std,3.1695652729656754,2.823360418014425,3.2943936868853716,2.3827500687396506,3.5039389780089056,2.4624944381244624,3.5475057030680706,2.4722954628911915,3.477552127922753,2.446264186757934,3.177221783362017,2.3734892271634775,2.8889224359449974,2.8748763608859234,2.881995415694215,2.7146615768940943,3.3542850735304626,2.166994250740668,3.441836201416299,2.18140478221384,3.393493656346563,2.176445519124465,3.157967428398117,2.658945133846355,2.8889224359449974,2.8748763608859234,3.0107225138118436,2.6680575838559313,3.3473285006648545,2.63863177494352,3.464364348833438,2.6667221602590248,3.3810549234628957,2.6327982755086006,3.1421419254133878,2.630005395328448,3.1695652729656754,2.823360418014425,3.157967428398117,2.658945133846355,3.3889220039656935,2.6458285496766325,3.4762560113234473,2.634693229127682,3.3403377977702693,2.653388638298678,2.881995415694215,2.7146615768940943,0.0,531.8839159064692
min,28.0,33.0,22.0,40.0,16.0,43.0,13.0,43.0,11.0,43.0,7.0,40.0,5.0,34.0,7.0,34.0,11.0,39.0,13.0,39.0,16.0,39.0,26.0,34.0,5.0,34.0,8.0,31.0,10.0,29.0,12.0,30.0,15.0,29.0,21.0,30.0,28.0,33.0,26.0,34.0,15.0,32.0,12.0,32.0,11.0,32.0,7.0,34.0,1.0,1.0
25%,45.0,46.0,40.0,52.0,34.0,54.0,29.0,54.0,25.0,54.0,19.0,51.0,15.0,45.0,17.0,45.0,25.0,50.0,29.0,50.0,34.0,50.0,43.0,46.0,15.0,45.0,20.0,43.0,26.0,42.0,30.0,43.0,34.0,43.0,40.0,44.0,45.0,46.0,43.0,46.0,34.0,45.0,29.0,46.0,25.0,45.0,17.0,45.0,1.0,461.25
50%,47.0,48.0,42.0,53.0,36.0,56.0,31.0,56.0,27.0,55.0,21.0,53.0,16.0,47.0,18.0,47.0,27.0,51.0,31.0,51.0,36.0,51.0,45.0,48.0,16.0,47.0,22.0,45.0,27.0,44.0,32.0,45.0,36.0,44.0,42.0,45.0,47.0,48.0,45.0,48.0,36.0,47.0,32.0,47.0,27.0,47.0,18.0,47.0,1.0,921.5
75%,49.0,49.0,44.0,55.0,38.0,57.0,33.0,57.0,29.0,57.0,23.0,54.0,18.0,48.75,20.0,49.0,29.0,52.0,33.0,53.0,38.0,52.0,47.0,50.0,18.0,48.75,24.0,46.0,29.0,46.0,34.0,47.0,38.0,46.0,43.0,47.0,49.0,49.0,47.0,50.0,38.0,49.0,34.0,49.0,29.0,48.0,20.0,49.0,1.0,1381.75
max,61.0,58.0,62.0,60.0,60.0,64.0,58.0,65.0,56.0,64.0,50.0,60.0,45.0,57.0,47.0,58.0,56.0,60.0,59.0,60.0,60.0,59.0,61.0,58.0,45.0,57.0,51.0,57.0,57.0,56.0,59.0,57.0,61.0,57.0,62.0,57.0,61.0,58.0,61.0,58.0,60.0,59.0,59.0,59.0,56.0,59.0,47.0,58.0,1.0,1842.0







#### DATA OVERVIEW INCLUDES NEUTRAL FACES
,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,class,file
count,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0,6599.0
mean,45.135778148204274,48.8477041976057,40.627973935444764,52.871041066828305,35.774056675253824,54.47446582815578,31.877254129413547,54.79815123503561,28.074253674799213,54.44688589180179,23.355963024700714,52.793605091680554,19.069252917108653,48.73980906197909,20.942718593726322,48.85346264585543,28.22139718139112,50.24367328383088,31.869828761933626,50.61782088195181,35.606152447340506,50.27110168207304,43.248370965297774,48.94938627064707,19.069252917108653,48.73980906197909,23.623579330201547,46.446128201242615,28.386270647067736,45.36081224427944,31.885740263676315,46.28246704046067,35.505379602970145,45.39066525231095,40.39521139566601,46.543112592817096,45.135778148204274,48.8477041976057,43.248370965297774,48.94938627064707,35.56099409001364,48.16896499469617,31.8843764206698,48.506440369752994,28.32217002576148,48.13153508107289,20.942718593726322,48.85346264585543,0.27913320200030306,3300.0
std,3.576733078930567,3.3348068998821847,3.7569714124444675,2.9097065543062177,3.952700768088797,2.9867710622008494,4.042750029756418,2.965220685550156,4.047527877441273,2.9249253780176074,3.933342036728916,2.850761296906736,3.69074589298317,3.4620501839397617,3.692285437682688,3.2370079436389303,3.748649119479481,2.7222705227382336,3.76936613811002,2.742681414871259,3.677912045281747,2.773231916158851,3.580620431959006,3.1229987836382875,3.69074589298317,3.4620501839397617,3.4771799924132996,3.15946301029456,3.5099217642624394,3.0539730592882015,3.584722393801261,3.093017918127514,3.466016523356151,3.0256773453114314,3.3740555846376146,3.071776607901143,3.576733078930567,3.3348068998821847,3.580620431959006,3.1229987836382875,3.5597613062617577,3.015485043928024,3.6585215752467426,3.0242396573929353,3.5976202628695444,3.0504952054921097,3.692285437682688,3.2370079436389303,0.44860712693509897,1905.1115452907213
min,28.0,33.0,22.0,33.0,16.0,33.0,13.0,33.0,11.0,33.0,7.0,34.0,5.0,34.0,7.0,33.0,11.0,31.0,13.0,31.0,16.0,30.0,26.0,33.0,5.0,34.0,8.0,29.0,10.0,27.0,12.0,28.0,15.0,27.0,21.0,29.0,28.0,33.0,26.0,33.0,15.0,30.0,12.0,30.0,11.0,30.0,7.0,33.0,0.0,1.0
25%,43.0,47.0,38.0,51.0,33.0,53.0,29.0,53.0,26.0,53.0,21.0,51.0,17.0,46.0,19.0,47.0,26.0,49.0,30.0,49.0,33.0,49.0,41.0,47.0,17.0,46.0,21.0,44.0,26.0,43.0,30.0,44.0,34.0,43.0,38.0,44.0,43.0,47.0,41.0,47.0,34.0,46.0,30.0,47.0,26.0,46.0,19.0,47.0,0.0,1650.5
50%,45.0,49.0,41.0,53.0,36.0,55.0,32.0,55.0,28.0,55.0,23.0,53.0,19.0,49.0,21.0,49.0,28.0,51.0,32.0,51.0,36.0,51.0,43.0,49.0,19.0,49.0,24.0,46.0,28.0,45.0,32.0,46.0,36.0,45.0,41.0,47.0,45.0,49.0,43.0,49.0,36.0,48.0,32.0,49.0,28.0,48.0,21.0,49.0,0.0,3300.0
75%,47.0,51.0,43.0,55.0,38.0,56.0,34.0,57.0,30.0,56.0,26.0,55.0,21.0,51.0,23.0,51.0,30.0,52.0,34.0,52.0,38.0,52.0,46.0,51.0,21.0,51.0,26.0,49.0,30.0,47.0,34.0,48.0,38.0,47.0,43.0,49.0,47.0,51.0,46.0,51.0,38.0,50.0,34.0,51.0,30.0,50.0,23.0,51.0,1.0,4949.5
max,61.0,60.0,62.0,62.0,60.0,65.0,58.0,65.0,56.0,65.0,50.0,62.0,45.0,59.0,47.0,59.0,56.0,61.0,59.0,61.0,60.0,61.0,61.0,59.0,45.0,59.0,51.0,57.0,57.0,56.0,59.0,57.0,61.0,57.0,62.0,58.0,61.0,60.0,61.0,59.0,60.0,59.0,59.0,59.0,56.0,59.0,47.0,59.0,1.0,6599.0
#### FEATURE MATRIX
I ran the script a couple times so all the data is the same.
              v1         v2         v3         v4         v5         v6         v7         v8  ...        v41        v42        v43        v44        v45        v46        v47        v48
Mean   45.135778  48.847704  40.627974  52.871041  35.774057  54.474466  31.877254  54.798151  ...  35.560994  48.168965  31.884376  48.506440  28.322170  48.131535  20.942719  48.853463
StDev   3.576462   3.334554   3.756687   2.909486   3.952401   2.986545   4.042444   2.964996  ...   3.559492   3.015257   3.658244   3.024011   3.597348   3.050264   3.692006   3.236763
###SAME DATA BUT IN CSV
name,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48
Mean,45.135778148204274,48.8477041976057,40.627973935444764,52.871041066828305,35.774056675253824,54.47446582815578,31.877254129413547,54.79815123503561,28.074253674799213,54.44688589180179,23.355963024700714,52.793605091680554,19.069252917108653,48.73980906197909,20.942718593726322,48.85346264585543,28.22139718139112,50.24367328383088,31.869828761933626,50.61782088195181,35.606152447340506,50.27110168207304,43.248370965297774,48.94938627064707,19.069252917108653,48.73980906197909,23.623579330201547,46.446128201242615,28.386270647067736,45.36081224427944,31.885740263676315,46.28246704046067,35.505379602970145,45.39066525231095,40.39521139566601,46.543112592817096,45.135778148204274,48.8477041976057,43.248370965297774,48.94938627064707,35.56099409001364,48.16896499469617,31.8843764206698,48.506440369752994,28.32217002576148,48.13153508107289,20.942718593726322,48.85346264585543
StDev,3.576462062974171,3.3345542151385232,3.756686739482326,2.9094860802346325,3.952401264336269,2.9865447488030648,4.042443702797875,2.964996005065976,4.047221188456022,2.924703750786497,3.9330439998248132,2.850545289230957,3.6904662380561977,3.4617878577272143,3.692005666101464,3.2367626693105525,3.7483650771134975,2.7220642510559503,3.7690805259757636,2.7424735966163776,3.6776333628000732,2.7730217830336032,3.5803491214504453,3.1227621480036567,3.6904662380561977,3.4617878577272143,3.4769165197861085,3.159223611695204,3.509655810728154,3.0537416538657007,3.584450772479112,3.092783554200895,3.465753896606537,3.025448083909545,3.373799925945461,3.0715438534694495,3.576462062974171,3.3345542151385232,3.5803491214504453,3.1227621480036567,3.5594915762892914,3.015256554816591,3.6582443620184657,3.0240105049275265,3.5973476642510116,3.050264063593421,3.692005666101464,3.2367626693105525


[2 rows x 48 columns]




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


# Acquire Data
current_path = os.getcwd()
file = 'models/halseytrainer-happy.csv'
data = pd.read_csv(file)


data['boxplot'].plot(kind='box')
plt.show()



# # print(data.columns)
# dataX = data.copy().drop(['class'], axis=1)
# dataY = data['class'].copy()
#
# featuresToScale = dataX.drop(['file'], axis=1).columns
#
# sX = pp.StandardScaler(copy=True)
# dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])
# scalingFactors = pd.DataFrame(data=[sX.mean_, sX.scale_], index=['Mean', 'StDev'], columns=featuresToScale)
#
# correlationMatrix = pd.DataFrame(data=[],index=dataX.columns,columns=dataX.columns)
# for i in dataX.columns:
#     for j in dataX.columns:
#         correlationMatrix.loc[i,j] = np.round(pearsonr(dataX.loc[:,i],dataX.loc[:,j])[0],2)
# count_classes = pd.value_counts(data['class'],sort=True).sort_index()
# ax = sns.barplot(x=count_classes.index, y=tuple(count_classes/len(data)))
#
# ax.set_title('Frequency Percentage by Class')
# ax.set_xlabel('class')
# ax.set_ylabel('Frequency Percentage')
# plt.show()


df = data.describe()
df.to_csv(r'happy-vector.csv', header=True, index=True, sep=',', mode='w')
# # print(scalingFactors)
