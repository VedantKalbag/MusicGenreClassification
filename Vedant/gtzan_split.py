import argparse
parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--path', type=str, default="../datasets/gtzan10sAug/", help='path to dataset')
config = parser.parse_args()
path = config.path

import pandas as pd
import numpy as np
import os, glob
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.utils import shuffle
from tqdm import tqdm

"""Split dataset as per https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split 
to remove artist bias"""

def readSplitFile(path):
    # Read filenames from the artist biased removed master files for each subset.
    # Input: path - path to master train test validate split files containing filenames
    # returns lines: Array of all the files in that subset
    with open(path) as f:
        lines = f.readlines()
        # lines = [line[:-1] for line in lines]
        lines = [line.split('/')[1] for line in lines]
        lines = [line.split('.')[0] + '.' + line.split('.')[1] for line in lines]
        return lines

def loadDF():
    # Load the npy file containing features and labels for all samples
    df = pd.DataFrame(np.load(f'{path}Final/features/a0.5_min1.0_max10.0_3.0s_block_0.5s_hop.npy', allow_pickle=True), columns=['melspec','label','filename'])
    # X = np.stack(df[['melspec']].values)
    X = np.stack(df[['melspec', 'filename']].values)
    y = df['label'].to_numpy()
    fnames = df['filename'].to_numpy()
    del df
    gc.collect()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    return X, y, fnames

def saveFnames(fnames):
    # Get filenames from master feature files and save them in small chunks while clearing memory
    # for faster processing. Can work as in index pre-processing.
    # Input: fnames - fullnames of files
    # output: saved file with trimmed names for easy sorting in the next step in the same order
    
    numFnames = len(fnames)
    names = np.array([])
    j = 0
    for i in np.arange(numFnames):
        filename = fnames[i]
        tmp = filename.split('.')
        names = np.append(names, tmp[0] + '.' + tmp[1])
        # print (i)
        i += 1
        if i % 10000 == 0 or i == numFnames-1:
            np.savetxt(f'{path}/Final/filenames/{j}.txt', names, fmt="%s")
            names = np.array([])
            j += 1


def loadDatasetNames():
    # Load dataset containing names saved in the aboe step
    path = f'{path}/Final/filenames/'
    numFiles = len(glob.glob(f'{path}*.txt'))
    names = np.array([])
    for j in np.arange(numFiles):
        fpath = f'{path}{j}.txt'
        namesFile = np.loadtxt(fpath, dtype='str')
        names = np.append(names, namesFile)
        # print (f'Done: {j}th file')
    return names


def saveSubset(X, y, fnames, splitType, savePath):
    # Save the extracted subset for easy loading in the model
    # Save fnames to index rest of the dataset
    np.savetxt(f'{savePath}{splitType}_shuffledNames.txt', fnames, fmt="%s")
    df=pd.DataFrame()
    df['melspec'] = list(X[:,0])
    df['label'] = y
    # Save shuffled fnames at a path
    for idx, row in tqdm(df.iterrows()):
        # print(f'{splitType}, {idx}')
        np.save(f'{savePath}{splitType}/{idx}.npy',row[['melspec','label']])

def filteredSubset(splitType, savePath):
    """
    Generate one subset from existing extracted features for a given splitType
    splitType = [train, test, val], one at a time
    """
    path = f'gtzan_split/{splitType}_filtered.txt'
    lines = readSplitFile(path)
    
    # Load npy file
    X, y, fnames = loadDF()
    
    # load filenames file each row (each 3s sample)
    # names = saveFnames(fnames)
    names = loadDatasetNames()
    subsetIndices = np.nonzero(np.in1d(names,lines))[0]
    
    # extract items with same fname
    X = X[subsetIndices]
    y = y[subsetIndices]
    fnames = fnames[subsetIndices]
    
    # Shuffle the order of melspec + label combination
    X, y, fnames = shuffle(X, y, fnames)

    # save them one by one in savepath
    saveSubset(X, y, fnames, splitType, savePath)

    # return X, y, fnames

def filteredDataset(savePath):
    # Wrapper to generate and save the whole dataset
    splitTypes = ['train', 'test', 'val']
    splitNames = np.array([])
    for splitType in splitTypes:
        filteredSubset(splitType, savePath)

if __name__ == '__main__':
    # splitType = 'train'
    savePath = f'{path}/Final/features/'
    filteredDataset(savePath)