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
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")
parser.add_argument('--balAug', type=bool, default=True, help="Use balanced train test split for all augmentations")
parser.add_argument('--AEFeatureSet', type=bool, default=False, help="True for generating corresponding clean feature set for autoencoders")
config = parser.parse_args()

datasetPath = os.path.join(config.datasetDir, config.datasetName)
# if not os.path.exists(datasetPath):
#     os.makedirs(datasetPath)

# if config.balAug == False:
#     audioPath = os.path.join(datasetPath, f'audio/a{config.a}_min{config.min_snr}_max{config.max_snr}')

# if config.balAug == True:
#     audioPath = os.path.join(datasetPath, 'audio')

# savePath = os.path.join(datasetPath, 'features')
# if not os.path.exists(savePath):
#     os.makedirs(savePath)

# if __name__ == '__main__':
#     # Path to save the final feature set file
#     featureSetName = os.path.join(savePath, f'a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}s_block_{config.hopLength}s_hop.npy')
    
#     if not os.path.exists(featureSetName):
#         # Generate feature dataset from audio dataset
#         genDataset(datasetPath, audioPath, savePath, config.blockLength, config.hopLength, config.a, config.min_snr, config.max_snr, config.balAug, config.AEFeatureSet)
    
#     elif config.overwrite == 'y':
#         # Generate feature dataset from audio dataset
#         genDataset(datasetPath, audioPath, savePath, config.blockLength, config.hopLength, config.a, config.min_snr, config.max_snr, config.balAug, config.AEFeatureSet)
    
#     else:
#         print ('Feature file already exists. Re-run the script with "--overwrite y" in the terminal to overwrite the existing dataset')