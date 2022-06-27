"""Split dataset as per https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split 
to remove artist bias"""

import pandas as pd
import numpy as np
import os, glob
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")
config = parser.parse_args()

def readSplitFile(path):
    """ Read filenames from the artist biased removed master files for train/ test/ val subset."""
    # Input: path - path to master train test validate split files containing filenames
    # returns lines: Array of all the files in that subset
    with open(path) as f:
        lines = f.readlines()
        # lines = [line[:-1] for line in lines]
        lines = [line.split('/')[1] for line in lines]
        lines = [line.split('.')[0] + '.' + line.split('.')[1] for line in lines]
        return lines

def loadDF(path):
    # Load the npy file containing features and labels for all samples
    df = pd.DataFrame(np.load(path, allow_pickle=True), columns=['melspec','label','filename'])
    # X = np.stack(df[['melspec']].values)
    X = np.stack(df[['melspec', 'filename']].values)
    y = df['label'].to_numpy()
    fnames = df['filename'].to_numpy()
    del df
    gc.collect()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)
    return X, y, fnames

def saveFnames(fnames, namesPath):
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
            np.savetxt(f'{namesPath}/{j}.txt', names, fmt="%s")
            names = np.array([])
            j += 1


def loadDatasetNames(path):
    """Load dataset containing names saved in the above step"""
    
    # path = path of saved files containing fnames in smaller files
    numFiles = len(glob.glob(f'{path}*.txt'))
    names = np.array([])
    for j in np.arange(numFiles):
        fpath = f'{path}{j}.txt'
        namesFile = np.loadtxt(fpath, dtype='str')
        names = np.append(names, namesFile)
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
        print(f'{splitType}, {idx}')
        tmp = f'{savePath}{splitType}/'
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        np.save(f'{savePath}{splitType}/{idx}.npy',row[['melspec','label']])

def filteredSubset(splitPath, splitType, savePath, dfPath, namesPath):
    """
    Generate one subset from existing extracted features for a given splitType
    splitType = [train, test, val], one at a time. splitPath = path of txt file containing the filenames in that subset
    """
    lines = readSplitFile(splitPath)
    # print (f'Final print: {len(lines)}{lines}')
    
    # Load npy file
    X, y, fnames = loadDF(dfPath)
    
    if not os.path.exists(namesPath) and len(glob.glob(namesPath + "*.txt")) == 0:
        os.makedirs(namesPath)
        names = saveFnames(fnames, namesPath)
    
    names = loadDatasetNames(namesPath)
    print ('Names: ', len(names))
    
    # Get subindices for all 3 splits within the dataset
    subsetIndices = np.nonzero(np.in1d(names,lines))[0]
    print ('Subset indices: ', len(subsetIndices))
    
    # extract items with same fname
    X = X[subsetIndices]
    y = y[subsetIndices]
    fnames = fnames[subsetIndices]
    
    # Shuffle the order of melspec + label combination
    X, y, fnames = shuffle(X, y, fnames)

    # save them one by one in savepath
    saveSubset(X, y, fnames, splitType, savePath)

    # return X, y, fnames

def filteredDataset(dfPath, savePath, namesPath):
    # Wrapper to generate and save the whole dataset
    splitTypes = ['train', 'test', 'val']
    splitNames = np.array([])
    for splitType in splitTypes:
        splitPath = f'gtzan_split/{splitType}_filtered.txt'
        filteredSubset(splitPath, splitType, savePath, dfPath, namesPath)

if __name__ == '__main__':
    datasetPath = config.datasetDir + config.datasetName + '/'
    print (datasetPath)
    
    # Path to load the feature extraction df
    dfPath = f'{datasetPath}features/a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}s_block_{config.hopLength}s_hop.npy'
    
    # Path to save final feature files split by subset: Saving function after every processing
    savePath = f'{datasetPath}features/'
    
    # Path to save final names of all files in chunk for faster processing: Input to saveFnames
    namesPath = f'{datasetPath}filenames/'

    filteredDataset(dfPath, savePath, namesPath)