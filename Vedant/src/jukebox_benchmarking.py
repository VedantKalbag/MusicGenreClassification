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
# parser.add_argument('--search_range', type=tuple_type, default=(100,4000,100), help='range to search for best threshold (start, stop, step)')
parser.add_argument('--v', type=int, default=0, help='verbosity')
parser.add_argument('--train_set', type=str, default='snr5')
# parser.add_argument('--eval_set', type=str, default='snr5')
parser.add_argument('--num_features', type=int, default=0)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--degree', type=int, default=3)
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--gamma', type=str, default='scale')
parser.add_argument('--coef0', type=float, default=0.0)
parser.add_argument('--hparams_optimise', type=int, default=0)

config = parser.parse_args()
print(config)
try:
    gamma = float(config.gamma)
except:
    gamma = config.gamma

v = config.v
if config.suffix != '':
    suffix = '_' + config.suffix
else:
    suffix = ''
import glob
import csv
import os
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Find numpy paths (and randomize to remove label ordering)
npy_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.train_set}/train/*.npy'))
# npy_paths2 = sorted(glob.glob('../../datasets/gtzan10sAug/datasets/jukebox_snr1/test/*.npy'))
# npy_paths = npy_paths1 + npy_paths2
# test_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.eval_set}/test/*.npy'))
# test_paths2 = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.eval_set}/val/*.npy'))
# test_paths = test_paths1 + test_paths2

random.seed(42)

# Load TRAINING data
def load_data(npy_paths,config):
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
    return X_flattened,y

def main(n_features=1000):
    X_flattened, y = load_data(npy_paths, config)
    X_df = pd.DataFrame(X_flattened)
    from sklearn.feature_selection import mutual_info_classif
    print("Running mutual information classifier")
    feature_scores = mutual_info_classif(X_df, y, random_state=0)
    high_score_features = []
    for score, f_name in sorted(zip(feature_scores, X_df.columns), reverse=True)[:n_features]:
        high_score_features.append(f_name)

    X_mic = X_df[high_score_features].to_numpy()
    # {'C': 36.27537294604771, 'class_weight': None, 'degree': 2.0706630521971743, 'gamma': 0.002333252359832339, 'kernel': 'rbf'}
    clf = make_pipeline(StandardScaler(), SVC(C=float(config.C), kernel=config.kernel, degree=float(config.degree), gamma=gamma, coef0=config.coef0))#.fit(X_mic, y)
    if v == 2:
        print("Running cross-validation")
    scores = cross_val_score(clf, X_mic, y, cv=10)
    if v >= 1 :
        print('{:.1f}% +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))

    if config.hparams_optimise:
        # Fitting model on all training data
        rand_list = {
                    'C': stats.uniform(1,100),#stats.expon(scale=100), 
                    'gamma': stats.expon(scale=.1),
                    'kernel': ['rbf','poly'], 
                    'class_weight':['balanced', None], 
                    'degree':stats.uniform(2, 10)
                    }
        if v == 1:
            print("Running hyperparameter optimisation - randomized search CV")
        scaler = StandardScaler()
        X_scaled=scaler.fit_transform(X_mic)
        clf2 = RandomizedSearchCV(SVC(), rand_list, n_jobs=4, random_state=42)
        search = clf2.fit(X_scaled,y)
        print(search.best_params_)

    model = clf.fit(X_mic,y)
    # save the model to disk
    output_path = '../models/jukebox/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path,f'model_{config.train_set}_{str(config.C)}C_{config.kernel}kernel_{str(config.degree)}degree_{config.gamma}gamma_{str(config.coef0)}coef0.sav')
    pickle.dump(model, open(filename, 'wb'))

    #TESTING
    for d in ['clean','snr1','snr5','snr10']:
        test_path = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{d}/test/*.npy'))
        X_test, y_test = load_data(test_path, config)
        X_test_df = pd.DataFrame(X_test)
        y_pred = model.predict(X_test_df[high_score_features].to_numpy())
        acc = accuracy_score(y_test, y_pred)
        if v == 1:
            print(f"Accuracy score on unseen {d} data: {acc*100}%")
        with open(os.path.join('..','logs', f"MIC_{str(n_features)}_features_{config.train_set}_train{suffix}.csv"),'a') as out_file: # need "a" and not w to append to a file, if not will overwrite
                        writer=csv.writer(out_file, delimiter='\t',lineterminator='\n',)
                        row=[n_features, d ,np.mean(scores),np.std(scores), acc]
                        writer.writerow(row)



if __name__=='__main__':
    main(config.num_features)