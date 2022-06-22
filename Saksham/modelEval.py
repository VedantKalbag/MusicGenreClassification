from util import *
import argparse
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
import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Xception', help='model')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use for training')
parser.add_argument('--multigpu', type=int, default=0, help='Train on multiple GPUs')
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--audioType', type=str, default = 'Augmented', help='Final augmented data')
parser.add_argument('--a', type=float, default = 0.5, help='Blend factor')
parser.add_argument('--min_snr', type=float, default = 1.0, help='Minimum snr for BG audio')
parser.add_argument('--max_snr', type=float, default = 10.0, help='Maximum snr for BG audio')
parser.add_argument('--batch', type=int, default = 32, help='Batch Size')
config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu
print("Train on multiple gpus: ", bool(config.multigpu))
print("Training on GPU",os.environ['CUDA_VISIBLE_DEVICES'])

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, mode, batch_size):
        self.batch_size = batch_size
        self.mode = mode
        if self.mode == 'train':
            self.data_path = '../datasets/gtzan10sAug/vedant/train'
        if self.mode == 'test':
            self.data_path = '../datasets/gtzan10sAug/vedant/test'
        if self.mode == 'val':
            self.data_path = '../datasets/gtzan10sAug/vedant/val'
        _,_,self.filenames = next(os.walk(self.data_path))

    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = tmp[0]
            y[i] = tmp[1]

        # bat = np.array([np.load(os.path.join(self.data_path, x), allow_pickle=True) for x in batch])
        return X,y #bat[:,0], bat[:,1]

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

a = config.a
min_snr = config.min_snr
max_snr = config.max_snr
batch_size = config.batch

modelName = f'model_10s_{config.model}_{config.blockLength}block_{config.hopLength}slide.h5'
modelPath = f'models/{modelName}'

# Load eval data
# folderDir = f'../datasets/gtzan10sAug/Final/features/'
# dataPath = f'a{a}_min{min_snr}_max{max_snr}_{config.blockLength}s_block_{config.hopLength}s_hop.npy'

savePath = '/models/'
saveName = f'Pred_a{a}_min{min_snr}_max{max_snr}_{config.blockLength}s_block_{config.hopLength}s_hop.csv'
# path = folderDir + dataPath

def gtruth(dataset):
    y = np.array([])
    for batch in dataset:
        y = np.append(y, batch[1])
    print ('Num Labels: ', len(y), y)
    return y

def main(modelPath, savePath, savefname, a, min_snr, max_snr, batch_size):
    # Load the model
    model = keras.models.load_model(modelPath)
    dataset = AugmentedDataset('train', batch_size)

    y = gtruth(dataset)

    y_pred = model.predict(dataset, batch_size = 32, verbose = 1)
    y_pred = np.argmax(y_pred, axis=-1)
    print ('Total predictions: ', y_pred.shape)
    np.savetxt('train_augmented_predictions.csv', y_pred)

    # Get confusion matrix
    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    print (f'f1 score: {f1}, avg: {np.mean(f1)}')
    print (f'recall: {recall}, avg: {np.mean(recall)}')
    accuracy = accuracy_score(y, y_pred)
    print (cm)
    print (f'accuracy: {accuracy}, avg: {np.mean(accuracy)}')


if __name__ == '__main__':
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    
    main(modelPath, savePath, saveName, a, min_snr, max_snr, batch_size)