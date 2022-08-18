import argparse
import os
import shutil
from util import *
# from augmentation import *
from augmentation import mainAugment
from featuresExtract import *
from gtzanSplit import *
import warnings

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
parser.add_argument('--dstype', type=str, default="normal", help="Type of dataset to generate: ['malspec', 'hpss', 'cqt', 'all']")

config = parser.parse_args()

warnings.filterwarnings("ignore")
datasetDir = config.datasetDir
datasetName = config.datasetName
a = config.a #0.5   # blend factor
min_snr = config.min_snr #1.0
max_snr = config.max_snr #10.0
nBG = config.nBG
nIR = config.nIR
overwrite = config.overwrite
blockLength = config.blockLength
hopLength = config.hopLength
dstype = config.dstype


def augmentation(datasetDir, datasetName, nBG, nIR, min_snr, max_snr, a, overwrite):
    output_path = os.path.join(datasetDir, datasetName)
    if not os.path.exists(output_path):
        print (f'path doesnt exist')
        os.makedirs(output_path)
        mainAugment(output_path, nBG, nIR, min_snr, max_snr, a)

    elif os.path.exists(output_path) and overwrite == 'y':
        shutil.rmtree(output_path)
        mainAugment(output_path, nBG, nIR, min_snr, max_snr, a)
    else:
        print ('Dataset file of same config already exists. Re-run the script with "--overwrite y" in the terminal to overwrite the existing dataset')

def extract_features(datasetDir, datasetName, a, min_snr, max_snr, blockLength, hopLength, dstype):
    datasetPath = os.path.join(datasetDir, datasetName)
    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    audioPath = os.path.join(datasetPath, 'audio')

    savePath = os.path.join(datasetPath, 'features')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print (savePath)

    # Path to save the final feature set file
    featureSetName = os.path.join(savePath, '*.npy')
    
    if not os.path.exists(featureSetName):
        # Generate feature dataset from audio dataset
        genDataset(datasetPath, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype)
    
    elif overwrite == 'y':
        # Generate feature dataset from audio dataset
        genDataset(datasetPath, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype)
    
    else:
        print ('Feature file already exists. Re-run the script with "--overwrite y" in the terminal to overwrite the existing dataset')


# def prepareFinalDataset(datasetDir, datasetName, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet):
#     datasetPath = os.path.join(datasetDir, datasetName)

#     balAugFinalDataset(datasetPath, a, min_snr, max_snr, blockLength, hopLength, AEFeatureSet)

if __name__ == '__main__':
    # Augment audio
    # augmentation(datasetDir, datasetName, nBG, nIR, min_snr, max_snr, a, overwrite)
    # Extract features
    print ('feature extraction started')
    extract_features(datasetDir, datasetName, a, min_snr, max_snr, blockLength, hopLength, dstype)
    # # Split as per gtzan
    # prepareFinalDataset(datasetDir, datasetName, a, min_snr, max_snr, blockLength, hopLength, dstype)