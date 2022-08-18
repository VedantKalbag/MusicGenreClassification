import argparse
import numpy as np
import pandas as pd
import os
import glob
import librosa
import matplotlib.pyplot as plt
import math
import soundfile as sf
from tqdm import tqdm
import os
from util import *
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser()
# parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
# parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
# parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
# parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
# parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
# parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
# parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
# parser.add_argument('--dstype', type=str, default="cqt", help="True for generating corresponding clean feature set for autoencoders")
# config = parser.parse_args()

# datasetPath = os.path.join(config.datasetDir, config.datasetName)

"""
Try extracting one .npy file per 10s track. This will help condense the code if we just append 
different slices of one track in one file and then save them individually. df creation will happen 
inside the function.
"""


def genCQT(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, audioType):
    
    splitTypes = ['train', 'test', 'val']

    for splitType in splitTypes:
        audioDir = os.path.join(audioPath, audioType, splitType)
        dirName, subdirList, _ = next(os.walk(audioDir))
        # Create list of files in each subdir
        df = pd.DataFrame(columns = ['cqt', 'label', 'filename'])
        # idx = 0
        for subdir in subdirList:
            _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
            print(f"Processing {subdir} files")
            for filename in tqdm(fileList):
                fname = filename.split('/')[-1]
                try:
                    fpath = os.path.join(audioDir, subdir, filename)
                    x, sr = librosa.load(fpath)
                    blockSamples = int(blockLength*sr)
                    hopSamples = int(hopLength*sr)
                    pin = 0
                    pend = len(x)-blockSamples
                    while pin <= pend:
                        chunk = x[pin:pin+blockSamples]
                        cqtChunk = librosa.cqt(chunk, sr=sr, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='constant', res_type=None, dtype=None)
                        if cqtChunk.shape[1] == int(blockLength/3*130):
                            df = df.append({'cqt' : cqtChunk, 'label' : subdir, 'filename': fname}, ignore_index = True)
                        pin += hopSamples
                except Exception as e:
                    print (e)
                    continue
        print (splitType, ' dataset: Completed!')
        print (df.count())
        out = df.to_numpy()
        np.save(os.path.join(savePath, f'{audioType}_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), out)

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
    np.savetxt(os.path.join(savePath, f'{splitType}_shuffledNames.txt'), fnames, fmt="%s")
    df=pd.DataFrame()
    df['cqt'] = list(X[:,0])
    df['label'] = y
    # Save shuffled fnames at a path
    for idx, row in tqdm(df.iterrows()):
        # print(f'{splitType}, {idx}')
        tmp = os.path.join(savePath, splitType)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        np.save(os.path.join(savePath, splitType, f'{idx}.npy'),row[['cqt','label']])

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

def saveSubSetWrapper(X, y, fnames, splitType, datasetPath, featureType):
    savePath = os.path.join(datasetPath, 'features', featureType)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveSubset(X, y, fnames, splitType, savePath)


def splitCQTDataset(datasetPath, a, min_snr, max_snr, blockLength, hopLength, featureType, audioType):
    """
    featureType: feature folder name
    audioType: cleanAudio/ augmentedAudioTrimmed
    """
    splitTypes = ['train', 'test', 'val']
    for splitType in splitTypes:
        # Load each subset
        dfPath = os.path.join(datasetPath, 'features', featureType, f'{audioType}_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy')
        X, y, fnames = loadDF(dfPath)
        # Save fnames
        saveSubSetWrapper(X, y, fnames, splitType, datasetPath, featureType)


### generate one .npy file per subset for each audioType

def genCQTDS(datasetPath, blockLength, hopLength, a, min_snr, max_snr):
    audioTypes = ['augmentedAudioTrimmed', 'cleanAudio']
    fpaths = ['featuresCQTAugmented', 'featuresCQTClean']
    audioPath = os.path.join(datasetPath, 'audio')

    for i, audioType in enumerate(audioTypes):
        print (audioType)
        print ('cqt')
        featureType = fpaths[i]
        savePath = os.path.join(datasetPath, 'features', featureType)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        genCQT(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, audioType)
        # Split the dataset
        splitCQTDataset(datasetPath, a, min_snr, max_snr, blockLength, hopLength, featureType, audioType)