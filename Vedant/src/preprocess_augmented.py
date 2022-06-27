import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import math
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, mixed_precision
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
import sys
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import json
import gc

# LOAD THE .NPY FILE CONTAINING ALL ROWS
df = pd.DataFrame(np.load('../../datasets/gtzan10sAug/vedant/Final/features/3.0s_block_0.5s_hop.npy', allow_pickle=True), columns=['melspec','label','filename'])
X = np.stack(df[['melspec','filename']].values)
y = df['label'].to_numpy()
names = df['filename'].to_numpy()
del df
gc.collect()

# ENCODE LABELS FROM 0-9
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# CREATE 70-15-15 TRAIN-TEST-VALIDATION SPLIT (RANDOM SPLIT, NOT DONE AT AN ARTIST LEVEL)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.50, shuffle=True, random_state=42)

# SAVE TRAIN DATA AS INDIVIDUAL ROWS
df=pd.DataFrame()
df['melspec'] = list(x_train[:,0])
df['label'] = y_train
df['filename'] = x_train[:,1]
for idx, row in tqdm(df.iterrows()):
    # filename = row['filename']
    np.save(f'../../datasets/gtzan10sAug/vedant/singleaug/train/{idx}.npy',row[['melspec','label']])

# SAVE TEST DATA AS INDIVIDUAL ROWS
df=pd.DataFrame()
df['melspec'] = list(x_test[:,0])
df['label'] = y_test
df['filename'] = x_test[:,1]
for idx, row in tqdm(df.iterrows()):
    # filename = row['filename']
    np.save(f'../../datasets/gtzan10sAug/vedant/singleaug/test/{idx}.npy',row[['melspec','label']])

# SAVE VAL DATA AS INDIVIDUAL ROWS
df=pd.DataFrame()
df['melspec'] = list(x_val[:,0])
df['label'] = y_val
df['filename'] = x_val[:,1]
for idx, row in tqdm(df.iterrows()):
    # filename = row['filename']
    np.save(f'../../datasets/gtzan10sAug/vedant/singleaug/val/{idx}.npy',row[['melspec','label']])