import argparse
from importlib.util import module_for_loader
from operator import truediv
parser = argparse.ArgumentParser()
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)
# Model configuration.
parser.add_argument('--aggregate',type=str, default='flatten', help='Method to convert from 2d array to 1d array')
parser.add_argument('--search_range', type=tuple_type, default=(100,4000,100), help='range to search for best threshold (start, stop, step)')
parser.add_argument('--v', type=int, default=0, help='verbosity')
parser.add_argument('--train_set', type=str, default='snr5')
parser.add_argument('--eval_set', type=str, default='snr5')
parser.add_argument('--num_features', type=int, default=0)
parser.add_argument('--suffix', type=str, default='')
config = parser.parse_args()
print(config)

v = config.v
if config.suffix != '':
    suffix = '_' + config.suffix
import glob
import csv
import os
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
if v == 1:
    print("---------------------------------------------------------------------------")
    print(f"Training and running cross validation on {config.train_set}")
    print(f"Evaluating final results on {config.eval_set}")
    print("---------------------------------------------------------------------------")

# Find numpy paths (and randomize to remove label ordering)
npy_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.train_set}/train/*.npy'))
# npy_paths2 = sorted(glob.glob('../../datasets/gtzan10sAug/datasets/jukebox_snr1/test/*.npy'))
# npy_paths = npy_paths1 + npy_paths2
test_paths1 = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.eval_set}/test/*.npy'))
test_paths2 = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.eval_set}/val/*.npy'))
test_paths = test_paths1 + test_paths2

# print(len(npy_paths))
# print(len(test_paths))
# print(f"There are {len(npy_paths)} files loaded")
random.seed(42)
random.shuffle(npy_paths)
random.shuffle(test_paths)

# Load TRAINING data
i=0
for p in npy_paths:
    if i == 0:
        T=pd.DataFrame(np.load(p, allow_pickle=True))
        i+=1
    else:
        tmp = pd.DataFrame(np.load(p,allow_pickle=True))
        T=pd.concat([T,tmp])
y = np.array(pd.DataFrame(T)[1].to_list())
X = np.array(pd.DataFrame(T)[0].to_list())

# LABEL ENCODING FOR GENRES
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
if v == 2:
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

# JUKEBOX LATENT VECTOR AGGREGATION FROM (N,4800) to (1,4800) for mean or (1,N*4800) for ravel
if config.aggregate == 'flatten':
    if v == 2:
        print("converting n-d arrays to 1d using ravel")
    X_flattened=np.zeros((len(X),len(X[0].ravel())))
    for i in range(len(X)):
        X_flattened[i] = X[i].ravel()#.mean(axis=1)#.ravel()
elif config.aggregate == 'mean':
    if v == 2:
        print("converting n-d arrays to 1d using mean")
    # print(X[0].mean(axis=0).shape)
    X_flattened=np.zeros((len(X),X[0].shape[1]))
    for i in range(len(X)):
        X_flattened[i] = X[i].mean(axis=0)#.ravel()
# print("Number of features available",X_flattened.shape)

# MUTUAL INFO SELECTION
X_df = pd.DataFrame(X_flattened)
from sklearn.feature_selection import mutual_info_classif
print("Running mutual information classifier")
feature_scores = mutual_info_classif(X_df, y, random_state=0)


# Load EVALUATION data
# T = np.array([np.load(p, allow_pickle=True) for p in test_paths])
i=0
for p in test_paths:
    if i == 0:
        T=pd.DataFrame(np.load(p, allow_pickle=True))
        i+=1
    else:
        tmp = pd.DataFrame(np.load(p,allow_pickle=True))
        T=pd.concat([T,tmp])
# print(T.shape)
y_test = np.array(pd.DataFrame(T)[1].to_list())
X = np.array(pd.DataFrame(T)[0].to_list())

from sklearn import preprocessing
y_test = le.fit_transform(y_test)
# le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print(le_name_mapping)

if config.aggregate == 'flatten':
    if v == 2:
        print("converting test n-d arrays to 1d")
    X_test=np.zeros((len(X),len(X[0].ravel())))
    for i in range(len(X)):
        X_test[i] = X[i].ravel()#.mean(axis=1)#.ravel()
elif config.aggregate == 'mean':
    if v == 2:
        print("converting test n-d arrays to 1d")
    X_test=np.zeros((len(X),X[0].shape[1]))
    for i in range(len(X)):
        X_test[i] = X[i].mean(axis=0)#.ravel()

X_test_df = pd.DataFrame(X_test)

if config.num_features == 0:
    for threshold in tqdm(range(config.search_range[0],config.search_range[1],config.search_range[2])):
        high_score_features = []
        for score, f_name in sorted(zip(feature_scores, X_df.columns), reverse=True)[:threshold]:
            high_score_features.append(f_name)

        X_mic = X_df[high_score_features].to_numpy()
        # print(X_mic.shape)
        # CROSS VALIDATION ON TRAINING SET
        # print("Fitting SVM on the training data")
        clf = make_pipeline(StandardScaler(), SVC())#.fit(X_mic, y)
        # print("Running cross-validation")
        scores = cross_val_score(clf, X_mic, y, cv=10)
        if v == 1:
            print('{:.1f}% +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))

        # Fitting model on all training data
        model = clf.fit(X_mic,y)
        # save the model to disk
        output_path = '../models/jukebox/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = os.path.join(output_path,f'model_{config.train_set}.sav')
        pickle.dump(model, open(filename, 'wb'))
        # print("Testing on unseen data")
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test_df[high_score_features].to_numpy())
        acc = accuracy_score(y_test, y_pred)
        if v == 1:
            print(f"Accuracy score on unseen data: {acc}")
        with open(os.path.join('..','logs', f"MIC_{str(config.search_range).replace(', ','_')}_{config.train_set}.csv"),'a') as out_file: # need "a" and not w to append to a file, if not will overwrite
                        writer=csv.writer(out_file, delimiter='\t',lineterminator='\n',)
                        row=[threshold, high_score_features,np.mean(scores),np.std(scores), acc]
                        writer.writerow(row)

else:
    threshold = config.num_features
    print(f"Running for {threshold} number of features")
    high_score_features = []
    for score, f_name in sorted(zip(feature_scores, X_df.columns), reverse=True)[:threshold]:
        high_score_features.append(f_name)

    X_mic = X_df[high_score_features].to_numpy()
    # print(X_mic.shape)
    # CROSS VALIDATION ON TRAINING SET
    # print("Fitting SVM on the training data")
    clf = make_pipeline(StandardScaler(), SVC())#.fit(X_mic, y)
    # print("Running cross-validation")
    scores = cross_val_score(clf, X_mic, y, cv=10)
    if v == 1:
        print('{:.1f}% +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))

    # Fitting model on all training data
    model = clf.fit(X_mic,y)
    # save the model to disk
    output_path = '../models/jukebox/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path,f'model_{config.train_set}.sav')
    pickle.dump(model, open(filename, 'wb'))
    # print("Testing on unseen data")
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test_df[high_score_features].to_numpy())
    acc = accuracy_score(y_test, y_pred)
    if v == 1:
        print(f"Accuracy score on unseen data: {acc*100}%")
    with open(os.path.join('..','logs', f"MIC_{str(threshold)}_features_{config.train_set}{suffix}.csv"),'a') as out_file: # need "a" and not w to append to a file, if not will overwrite
                    writer=csv.writer(out_file, delimiter='\t',lineterminator='\n',)
                    row=[threshold, high_score_features,np.mean(scores),np.std(scores), acc]
                    writer.writerow(row)