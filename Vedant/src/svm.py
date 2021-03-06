import argparse
from operator import truediv
parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--aggregate',type=str, default='flatten', help='Method to convert from 2d array to 1d array')
config = parser.parse_args()
print(config)

import glob
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Find numpy paths (and randomize to remove label ordering)
npy_paths1 = sorted(glob.glob('../../datasets/gtzan10sAug/datasets/jukebox/train/*.npy'))
npy_paths2 = sorted(glob.glob('../../datasets/gtzan10sAug/datasets/jukebox/test/*.npy'))
npy_paths = npy_paths1 + npy_paths2
print(f"There are {len(npy_paths)} files loaded")
# assert len(npy_paths) == 1000
random.seed(42)
random.shuffle(npy_paths)

# Load data
T = np.array([np.load(p, allow_pickle=True) for p in npy_paths])
y = np.array(pd.DataFrame(T)[1].to_list())
X = np.array(pd.DataFrame(T)[0].to_list())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# X = np.asarray(X)
# Y = np.asarray(y)
if config.aggregate == 'flatten':
    print("converting n-d arrays to 1d using ravel")
    X_flattened=np.zeros((len(X),len(X[0].ravel())))
    for i in tqdm(range(len(X))):
        X_flattened[i] = X[i].ravel()#.mean(axis=1)#.ravel()
elif config.aggregate == 'mean':
    print("converting n-d arrays to 1d using mean")
    print(X[0].mean(axis=0).shape)
    X_flattened=np.zeros((len(X),X[0].shape[1]))
    for i in tqdm(range(len(X))):
        X_flattened[i] = X[i].mean(axis=0)#.ravel()

print("Number of features available",X_flattened.shape)

# MUTUAL INFO SELECTION
X_df = pd.DataFrame(X_flattened)
from sklearn.feature_selection import mutual_info_classif
threshold = 1000  # the number of most relevant features
print(f"Running Mutual information selection for {threshold} best features")
high_score_features = []
feature_scores = mutual_info_classif(X_df, y, random_state=0)
for score, f_name in sorted(zip(feature_scores, X_df.columns), reverse=True)[:threshold]:
        # print(f_name, score)
        high_score_features.append(f_name)
X_mic = X_df[high_score_features].to_numpy()
# print(high_score_features)





# CROSS VALIDATION ON TRAINING SET
print("Fitting SVM on the training data")
clf = make_pipeline(StandardScaler(), SVC()).fit(X_mic, y)
print("Running cross-validation")
scores = cross_val_score(clf, X_mic, y, cv=10)
print('{:.1f} +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))

print("Testing on unseen data")
# VALIDATION SET
test_paths = sorted(glob.glob('../../datasets/gtzan10sAug/datasets/jukebox/val/*.npy'))
# assert len(npy_paths) == 1000
random.seed(42)
random.shuffle(test_paths)

# Load data
T = np.array([np.load(p, allow_pickle=True) for p in test_paths])
print(T.shape)
y = np.array(pd.DataFrame(T)[1].to_list())
X = np.array(pd.DataFrame(T)[0].to_list())

from sklearn import preprocessing
y_test = le.fit_transform(y)
# le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print(le_name_mapping)

if config.aggregate == 'flatten':
    print("converting test n-d arrays to 1d")
    X_test=np.zeros((len(X),len(X[0].ravel())))
    for i in tqdm(range(len(X))):
        X_test[i] = X[i].ravel()#.mean(axis=1)#.ravel()
elif config.aggregate == 'mean':
    print("converting test n-d arrays to 1d")
    print(X[0].mean(axis=0).shape)
    X_test=np.zeros((len(X),X[0].shape[1]))
    for i in tqdm(range(len(X))):
        X_test[i] = X[i].mean(axis=0)#.ravel()

X_test_df = pd.DataFrame(X_test)
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test_df[high_score_features].to_numpy())
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score on unseen data: {acc}")