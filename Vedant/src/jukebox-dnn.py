import argparse
from importlib.util import module_for_loader
from operator import truediv
parser = argparse.ArgumentParser()
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)
# Model configuration.
parser.add_argument('--aggregate',type=str, default='mean', help='Method to convert from 2d array to 1d array')
# parser.add_argument('--search_range', type=tuple_type, default=(100,4000,100), help='range to search for best threshold (start, stop, step)')
parser.add_argument('--v', type=int, default=1, help='verbosity')
parser.add_argument('--train_set', type=str, default='snr5')
parser.add_argument('--gpu', type=str, default = '2', help='GPU to be used')
parser.add_argument('--suffix', type=str, default = '', help='save suffix')
parser.add_argument('--lr',type=float, default = 1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
config = parser.parse_args()

if config.v >= 1 :
    print(config)

import glob
import csv
import os
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import sys
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, mixed_precision
from tensorflow.keras.utils import plot_model
from keras import backend as K

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu

npy_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.train_set}/train/*.npy'))
val_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.train_set}/val/*.npy'))
test_paths = sorted(glob.glob(f'../../datasets/jukebox/jukebox_{config.train_set}/test/*.npy'))
random.seed(42)

cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    mode='max',
    min_delta=0.01,
    patience=10,
    verbose=1
)
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False
)
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_acc',
    mode='max',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=5e-4
)
v = config.v
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

def build_model(n_features):
    if v >=1 :
        print("Building model")
    inputs = layers.Input(shape=(n_features))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def plot_history(number, history):
    with open('history'+str(number)+'.json', 'w') as fp:
        json.dump(history.history, fp, indent=4)

    fig, ax = plt.subplots(2, figsize=(10,8))

    # create accuracy subplot
    ax[0].plot(history.history['acc'], label='train accuracy')
    ax[0].plot(history.history['val_acc'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    # create error subplot
    ax[1].plot(history.history['loss'], label='train error')
    ax[1].plot(history.history['val_loss'], label='test error')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Loss eval')
    try:
        plt.savefig('history' + str(number) +  '.png')
    except:
        pass
    if v >=1:
        plt.show()

def main(n_features=1000, num_epochs = 50, modelname="test"):
    X_flattened, y_train = load_data(npy_paths, config)
    X_df = pd.DataFrame(X_flattened)
    X_val, y_val = load_data(val_paths, config)
    X_test,y_test = load_data(test_paths, config)
    from sklearn.feature_selection import mutual_info_classif
    print("Running mutual information classifier")
    feature_scores = mutual_info_classif(X_df, y_train, random_state=0)
    high_score_features = []
    for score, f_name in sorted(zip(feature_scores, X_df.columns), reverse=True)[:n_features]:
        high_score_features.append(f_name)

    X_train = X_df[high_score_features].to_numpy()
    X_val = pd.DataFrame(X_val)
    X_val = X_val[high_score_features].to_numpy()
    X_test = pd.DataFrame(X_test)
    X_test = X_test[high_score_features].to_numpy()
    if v >= 1:
        print(X_train.shape)

    # saver = CustomSaver()

    model = build_model(n_features)
    if v >= 1:
        print(model.summary())
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss='sparse_categorical_crossentropy',   metrics=['acc'])
    try:
        if v >=1 :
            print("Beginning training")
        history = model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), epochs=num_epochs, callbacks=[cb_earlystop, cb_csvlogger, cb_reducelr])
        test_acc = model.evaluate(x=X_test, y=y_test, return_dict=True)
        history.history['test_acc'] = test_acc['acc']
        history.history['test_loss'] = test_acc['loss']
        plot_history(modelname, history)
        model.save(f'./model_{modelname}_{num_epochs}_{config.suffix}.h5')
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, saving model")
        model.save(f'./model_{modelname}_{num_epochs}_incomplete.h5')
        sys.exit()
    #plot_model(model, show_shapes=True, to_file='model' + str(i) + '.png')
    K.clear_session()

if __name__ == '__main__':
    main(1000, config.epochs, 'jukebox_dnn_test')