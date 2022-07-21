import argparse
import os
import shutil
from util import *
from augmentation import *
from featuresExtract import *
from gtzanSplit import *
import warnings
# import numpy as np
# import pandas as pd
# import glob
# import librosa
# import soundfile as sf
# from tqdm import tqdm
# import random
# import audiomentations
# from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, ApplyImpulseResponse

parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
parser.add_argument('--nBG', type=int, default=1, help='Number of BG noise samples applied to each sample')
parser.add_argument('--nIR', type=int, default=1, help='Number of IR noise samples applied to each sample')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Output path with all versions of datasets")
parser.add_argument('--datasetName', type=str, default="newDir", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")
parser.add_argument('--balAug', type=bool, default=True, help="Use balanced train test split for all augmentations")
parser.add_argument('--AEFeatureSet', type=bool, default=True, help="True for generating corresponding clean feature set for autoencoders")

config = parser.parse_args()

warnings.filterwarnings("ignore")
datasetDir = config.datasetDir
datasetName = config.datasetName
balAug = config.balAug
a = config.a #0.5   # blend factor
min_snr = config.min_snr#1.0
max_snr = config.max_snr#10.0
audioPath = f'../datasets/gtzan10s/'    # Base audio to augment - 10s extracted from 30s gtzan clips
irPath = '../datasets/IR_MIT/'  # IR folder
bgPath = '../datasets/TAU/audio/'   # Background folder
nBG = config.nBG
nIR = config.nIR
overwrite = config.overwrite
AEFeatureSet = config.AEFeatureSet
blockLength = config.blockLength
hopLength = config.hopLength

def augmentWrapper(audioPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AEFeatureSet):
    if balAug == True:
        dfPath = output_path
        mainAugment(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AEFeatureSet)
    if balAug == False:    
        dfPath = os.path.join(output_path, 'AugDatasetDesc.csv')
        if not os.path.exists(output_path):
            mainAugment(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AEFeatureSet)

def augmentation(datasetDir, datasetName, balAug, audioPath, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, overwrite, AEFeatureSet):
    output_path = os.path.join(datasetDir, datasetName)
    if not os.path.exists(output_path):
        print (f'path doesnt exist')
        os.makedirs(output_path)
        augmentWrapper(audioPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AEFeatureSet)

    elif os.path.exists(output_path) and overwrite == 'y':
        shutil.rmtree(output_path)
        augmentWrapper(audioPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AEFeatureSet)
    else:
        print ('Dataset file of same config already exists. Re-run the script with "--overwrite y" in the terminal to overwrite the existing dataset')

def extract_features(datasetDir, datasetName, balAug, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet):
    datasetPath = os.path.join(datasetDir, datasetName)
    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    if balAug == False:
        audioPath = os.path.join(datasetPath, f'audio/a{a}_min{min_snr}_max{max_snr}')

    if balAug == True:
        audioPath = os.path.join(datasetPath, 'audio')

    savePath = os.path.join(datasetPath, 'features')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print (savePath)

    # Path to save the final feature set file
    featureSetName = os.path.join(savePath, '*.npy')
    
    if not os.path.exists(featureSetName):
        # Generate feature dataset from audio dataset
        genDataset(datasetPath, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, balAug, AEFeatureSet)
    
    elif overwrite == 'y':
        # Generate feature dataset from audio dataset
        genDataset(datasetPath, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, balAug, AEFeatureSet)
    
    else:
        print ('Feature file already exists. Re-run the script with "--overwrite y" in the terminal to overwrite the existing dataset')


def prepareFinalDataset(datasetDir, datasetName, balAug, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet):
    datasetPath = os.path.join(datasetDir, datasetName)

    if balAug == True:
        balAugFinalDataset(datasetPath, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet)

    if balAug == False:
        # Path to load the feature extraction df
        dfPath = os.path.join(datasetPath, f'features/a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy')
        # Path to save final feature files split by subset: Saving function after every processing
        savePath = os.path.join(datasetPath, 'features')
        # Path to save final names of all files in chunk for faster processing: Input to saveFnames
        namesPath = os.path.join(datasetPath, 'filenames')
        # namesPath = f'{datasetPath}filenames/'
        filteredDataset(dfPath, savePath, namesPath)


if __name__ == '__main__':
    # Augment audio
    augmentation(datasetDir, datasetName, balAug, audioPath, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, overwrite, AEFeatureSet)
    # Extract features
    extract_features(datasetDir, datasetName, balAug, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet)
    # Split as per gtzan
    prepareFinalDataset(datasetDir, datasetName, balAug, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet)