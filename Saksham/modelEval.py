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
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15, help='num epochs')
parser.add_argument('--model', type=str, default='Xception', help='model')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use for training')
parser.add_argument('--multigpu', type=int, default=0, help='Train on multiple GPUs')
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--audioType', type=str, default = 'Augmented', help='Final augmented data')
parser.add_argument('--a', type=float, default = 0.5, help='Blend factor')
parser.add_argument('--min_snr', type=float, default = 1.0, help='Minimum snr for BG audio')
parser.add_argument('--max_snr', type=float, default = 10.0, help='Maximum snr for BG audio')
parser.add_argument('--batch', type=int, default = 256, help='Batch Size')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu
print("Train on multiple gpus: ", bool(config.multigpu))
print("Training on GPU",os.environ['CUDA_VISIBLE_DEVICES'])

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, datasetName, datasetDir, batch_size):
        self.batch_size = batch_size
        self.datasetDir = datasetDir
        self.datasetName = datasetName
        self.data_path = f'{self.datasetDir}{self.datasetName}/features/val/'
        _,_,self.filenames = next(os.walk(self.data_path))

    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        IDs = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = tmp[0]
            y[i] = tmp[1]
            IDs[i] = ID.split('.')[0]
        # bat = np.array([np.load(os.path.join(self.data_path, x), allow_pickle=True) for x in batch])
        return X,y,IDs #bat[:,0], bat[:,1]

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

a = config.a
min_snr = config.min_snr
max_snr = config.max_snr
batch_size = config.batch
audioType = config.audioType
model = config.model
datasetName = config.datasetName
datasetDir = config.datasetDir
epochs = config.epochs

def gtruth(dataset):
    y = np.array([])
    for batch in dataset:
        y = np.append(y, batch[1])
    print ('Num Labels: ', len(y), y)
    return y

def getEvalSeq(dataset):
    seq = np.array([])
    for batch in dataset:
        seq = np.append(seq, batch[2])
    return seq

def saveCM(ConfusionMatrix, mapping, savePath):
    ConfusionMatrix = pd.DataFrame(ConfusionMatrix)
    ConfusionMatrix = ConfusionMatrix.rename(index = mapping, columns = mapping)

    plt.rcParams['figure.figsize']=[15,5]
    sns.set(font_scale=1.4)
    sns_plot = sns.heatmap(ConfusionMatrix, cmap="YlGnBu", annot=True, annot_kws={"size":16});
    fig = sns_plot.get_figure()
    fig.savefig(savePath + ".png")

def readFnames(datasetName, splitType):
    path = f'../datasets/gtzan10sAug/datasets/{datasetName}/features/{splitType}_shuffledNames.txt'
    with open(path) as f:
        lines = f.readlines()
        lines = [line.split('.')[0] + '.' + line.split('.')[1] + '.' + line.split('.')[2][0] for line in lines]
        fileNames = [line.split('.')[0] + '.' + line.split('.')[1] for line in lines]
    return np.array(lines), np.array(fileNames)

def eval10s(datasetName, splitType, valSeq, y, y_pred):
    # Get array of ordered filenames
    names,_ = readFnames(datasetName, splitType)

    # From valSeq array, aggregate labels for same fnames
    uniqueVals = np.unique(names)
    
    acc = np.array([])

    for tmp in uniqueVals:
        # find all indices containing tmp
        indices = np.where(names == tmp)[0]
        print (indices)
        # get y and y_pred for each file
        ytmp = y[indices]
        y_predtmp = y_pred[indices]
        # calculate accuracy between y and y_pred
        accuracy = accuracy_score(ytmp, y_predtmp)
        acc = np.append(acc, accuracy)
    return acc

def evalTrack(datasetName, splitType, valSeq):
    _, names = readFnames(datasetName, splitType)


def main(modelPath, savePath, a, min_snr, max_snr, batch_size, datasetDir, datasetName, mapping):
    # Load the model
    model = keras.models.load_model(modelPath)
    
    # Load dataset
    dataset = AugmentedDataset(datasetName, datasetDir, batch_size)

    y = gtruth(dataset)
    sequence = getEvalSeq(dataset)

    y_pred = model.predict(dataset, batch_size = 32, verbose = 1)
    y_pred = np.argmax(y_pred, axis=-1)
    # # print ('Total predictions: ', y_pred.shape)
    # np.savetxt(savePath + ".csv", y_pred)

    accuracies = eval10s(datasetName, 'val', sequence, y, y_pred)

    # # Get confusion matrix
    # cm = confusion_matrix(y, y_pred)
    # saveCM(cm, mapping, savePath)
    # f1 = f1_score(y, y_pred, average='weighted')
    # recall = recall_score(y, y_pred, average='weighted')
    # print (f'f1 score: {f1}, avg: {np.mean(f1)}')
    # print (f'recall: {recall}, avg: {np.mean(recall)}')
    # accuracy = accuracy_score(y, y_pred)
    # print (cm)
    # print (f'accuracy: {accuracy}, avg: {np.mean(accuracy)}')


if __name__ == '__main__':
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    modelPath = f'models/model_10s_{audioType}_{model}_a{a}_min{min_snr}_max{max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.h5'
    # Path to save predictions
    savePath = f'results/Pred_{audioType}train_{datasetName}eval_{model}_a{a}_min{min_snr}_max{max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs'
    
    mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    
    main(modelPath, savePath, a, min_snr, max_snr, batch_size, datasetDir, datasetName, mapping)
    
    