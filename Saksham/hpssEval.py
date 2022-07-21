from util import *
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='num epochs')
parser.add_argument('--model', type=str, default='CNN', help='model')
parser.add_argument('--multigpu', type=int, default=0, help='Train on multiple GPUs')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use for training')
parser.add_argument('--log_step', type=int, default=10, help='log interval')
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--audioType', type=str, default = 'Augmented', help='Slide duration')
parser.add_argument('--a', type=float, default = 0.5, help='Blend factor')
parser.add_argument('--min_snr', type=float, default = 1.0, help='Minimum snr for BG audio')
parser.add_argument('--max_snr', type=float, default = 5.0, help='Maximum snr for BG audio')
parser.add_argument('--batch', type=int, default = 256, help='Batch Size')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--balAug', type=bool, default=True, help="Use balanced train test split for all augmentations")
parser.add_argument('--AEFeatureSet', type=bool, default=False, help="True for generating corresponding clean feature set for autoencoders")
parser.add_argument('--cleanTrain', type=bool, default=False, help="Type of data for training: clean/augmented")
parser.add_argument('--cleanVal', type=bool, default=False, help="Type of data for training: clean/augmented")
parser.add_argument('--dstype', type=str, default='harmonic', help="Type of dataset: Harmonic/ Percussive")
parser.add_argument('--evalDtype', type=str, default='clean', help="Type of dataset: Harmonic/ Percussive")

config = parser.parse_args()

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, batch_size, path, dstype, cleanTrain, cleanVal, evalDtype):
        # Choose dstype as harmonic/ percussive and cleanVal T/F for clean or aug validation set
        self.batch_size = batch_size
        self.path = path
        self.cleanTrain = cleanTrain
        self.cleanVal = cleanVal
        self.dstype = dstype
        self.evalDtype = evalDtype

        if dstype == 'harmonic':
            if self.evalDtype == 'augmented':
                self.data_path = os.path.join(path, 'featuresAugH', 'val')
            if self.evalDtype == 'clean':
                self.data_path = os.path.join(path, 'featuresCleanH', 'val')

        if dstype == 'percussive':
            if self.evalDtype == 'augmented':
                self.data_path = os.path.join(path, 'featuresAugP', 'val')
            if self.evalDtype == 'clean':
                self.data_path = os.path.join(path, 'featuresCleanP', 'val')
        
        _,_,self.filenames = next(os.walk(self.data_path))
        
    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = np.abs(tmp[0])
            y[i] = tmp[1]

        return X,y #bat[:,0], bat[:,1]


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
        print ('batch: ', type(batch))
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


def main(modelPath, savePath, a, min_snr, max_snr, batch_size, datasetDir, datasetName, mapping, path, dstype, cleanTrain, cleanVal, evalDtype):
    # Load the model
    model = keras.models.load_model(modelPath)
    
    # Load dataset
    dataset = AugmentedDataset(batch_size, path, dstype, cleanTrain, cleanVal, evalDtype)

    y = gtruth(dataset)
    # sequence = getEvalSeq(dataset)

    y_pred = model.predict(dataset, batch_size = 32, verbose = 1)
    y_pred = np.argmax(y_pred, axis=-1)
    # # print ('Total predictions: ', y_pred.shape)
    # np.savetxt(savePath + ".csv", y_pred)

    # accuracies = eval10s(datasetName, 'val', sequence, y, y_pred)

    # Get confusion matrix
    cm = confusion_matrix(y, y_pred)
    saveCM(cm, mapping, savePath)
    f1 = f1_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    print (f'f1 score: {f1}, avg: {np.mean(f1)}')
    print (f'recall: {recall}, avg: {np.mean(recall)}')
    accuracy = accuracy_score(y, y_pred)
    print (cm)
    print (f'accuracy: {accuracy}, avg: {np.mean(accuracy)}')


if __name__ == '__main__':
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    modelPath = f'models/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{config.model}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.h5'
    # Path to save predictions
    savePath = f'results/Pred_{audioType}train_{datasetName}eval_{model}_a{a}_min{min_snr}_max{max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs'
    
    mapping = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    datasetPath = os.path.join(datasetDir, datasetName)
    path = os.path.join(datasetPath, 'features')
    print ('cleanTrain: ', config.cleanTrain)
    print ('cleanVal: ', config.cleanVal)
    print ('evalDtype: ', config.evalDtype)
    print ('modelname: ', config.model)
    main(modelPath, savePath, a, min_snr, max_snr, batch_size, datasetDir, datasetName, mapping, path, config.dstype, config.cleanTrain, config.cleanVal, config.evalDtype)