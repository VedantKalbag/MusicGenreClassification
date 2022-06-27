import argparse
import numpy as np
import pandas as pd
import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm
import random
import shutil
import audiomentations
from util import *
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, ApplyImpulseResponse
parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
parser.add_argument('--nBG', type=int, default=1, help='Number of BG noise samples applied to each sample')
parser.add_argument('--nIR', type=int, default=1, help='Number of IR noise samples applied to each sample')
parser.add_argument('--save_path', type=str, default="../datasets/gtzan10sAug/datasets/", help="Output path with all versions of datasets")
parser.add_argument('--dir_name', type=str, default="newDir", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")

import warnings
warnings.filterwarnings("ignore")

config = parser.parse_args()
print(config)
a = config.a#0.5   # blend factor
min_snr = config.min_snr#1.0
max_snr = config.max_snr#10.0
audioPath = f'../datasets/gtzan10s/'    # Base audio to augment - 10s extracted from 30s gtzan clips
irPath = '../datasets/IR_MIT/'  # IR folder
bgPath = '../datasets/TAU/audio/'   # Background folder

# Define number of BGs and IRs to be convolved with each 10s file
nBG = config.nBG#5
nIR = config.nIR#3

output_path = config.save_path + config.dir_name + '/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Total num of audio files - 44946
# Total num of BG files - 14400

def readNoisePaths(path, n):
    """Read n audios in an array. path: directory containing audio files"""

    allPaths = glob.glob(path + "*.wav")
    nPaths = random.sample(allPaths, n)

    return nPaths

def addBG(x, sr, noisePath, min_snr, max_snr):
    """Add background noise at noisePath with min and max snr in dB"""
    # Extract bg name
    bgName = noisePath.split('/')[-1].split('.')[0]
    augmentBG = Compose([
    AddBackgroundNoise(
        sounds_path= noisePath,
        min_snr_in_db=min_snr,
        max_snr_in_db=max_snr,
        noise_rms="relative",
        p=1,
        lru_cache_size=2)
    ])
    augmentBGAudio = augmentBG(samples=x, sample_rate=sr)
    return augmentBGAudio, bgName

def addIR(x, IRnoisePath, a):
    """Convolve x with IR located at IRnoisePath with a blend factor of a"""
    irname = IRnoisePath.split('/')[-1].split('.')[0]
    ir, fs = librosa.load(IRnoisePath, sr=22050)
    ##### Convolve audio sample with the IR
    conv = blendedConv(x, ir, a)
    return conv, irname

def genDatasetDF(audioPath, dfPath, bgPath, irPath, nBG, nIR):
    df = pd.DataFrame(columns = ['audioPath', 'IRPath', 'BGPath', 'fname', 'label'])
    dirName, subdirList, _ = next(os.walk(audioPath))
    i = 0
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        print(subdir)
        for filename in tqdm(fileList):
            path = os.path.join(dirName, subdir, filename)
            fname = filename.split('/')[-1][:-4]
            nBGPaths = readNoisePaths(bgPath, nBG)
            for noisePath in nBGPaths:
                nIRPaths = readNoisePaths(irPath, nIR)
                for IRnoisePath in nIRPaths:
                    df = df.append({'audioPath' : path, 'IRPath' : IRnoisePath, 'BGPath' : noisePath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df.to_csv(f'{dfPath}', index = False)
    return df

def augmentAudio(dfPath, output_path, min_snr, max_snr, a):
    print(dfPath)
    df = pd.read_csv(dfPath)
    audioPaths = df['audioPath'].tolist()
    IRPaths = df['IRPath'].tolist()
    BGPaths = df['BGPath'].tolist()
    fnames = df['fname'].tolist()
    labels = df['label'].tolist()
    for i in tqdm(np.arange(len(audioPaths))):
        # Read audio
        x, sr = librosa.load(audioPaths[i])

        # Convolve with IR
        conv, irname = addIR(x, IRPaths[i], a)

        # Add BG noise
        augmentBGAudio, bgname = addBG(conv, sr, BGPaths[i], min_snr, max_snr)
        
        finalSavePath = f'{output_path}/audio/a{a}_min{min_snr}_max{max_snr}/{labels[i]}/'
        if not os.path.exists(finalSavePath):
            os.makedirs(finalSavePath)
        finalFileName = f'{fnames[i]}_{bgname}_{irname}.wav'
        sf.write(f'{finalSavePath}{finalFileName}', augmentBGAudio, sr, subtype='PCM_24')
        i += 1

def main(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a):
    print ('Generating dataset file...')
    df = genDatasetDF(audioPath, dfPath, bgPath, irPath, nBG, nIR)
    print ('Dataset file generated! Generating dataset now')
    augmentAudio(dfPath, output_path, min_snr, max_snr, a)
    print ('Successfully generated!')

if __name__ == '__main__':
    dfPath = os.path.join(output_path, 'AugDatasetDesc.csv')
    if not os.path.exists(dfPath):
        main(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a)
    elif config.overwrite == 'y':
        shutil.rmtree(output_path)
        main(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a)
    else:
        print ('Dataset file of same config already exists. Re-run the script with --overwrite y in the terminal to overwrite the existing dataset')