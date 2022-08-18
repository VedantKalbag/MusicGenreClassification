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
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Output path with all versions of datasets")
parser.add_argument('--datasetName', type=str, default="newDir", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")
parser.add_argument('--dstype', type=str, default="normal", help="Type of dataset to generate: ['normal', 'hpss', 'autoencoder', 'all']")

import warnings
warnings.filterwarnings("ignore")

# Total num of audio files - 44946
# Total num of BG files - 14400

def addBG(x, sr, noisePath, min_snr, max_snr):
    """Add background noise at noisePath with min and max snr in dB"""
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

def readGtzanSubset(splitType):
    # Read split files
    splitPath = os.path.join('gtzan_split', f'{splitType}_filtered.txt')
    lines = readSplitFile(splitPath)
    return lines

def readNoisePaths(path, n, augmentType, pathType='dirPath'):
    """Read n audios in an array. path: directory containing audio files"""

    if pathType == 'dirPath':
        allPaths = glob.glob(path + "*.wav")
        nPaths = random.sample(allPaths, n)
    
    if pathType == 'array':
        nPaths = random.sample(path, n)
        if augmentType == 'BG':
            nPaths = [os.path.join('..','datasets','TAU','audio', nPath) for nPath in nPaths]
        if augmentType == 'IR':
            nPaths = [os.path.join('..','datasets','IR_MIT', nPath) for nPath in nPaths]

    return nPaths

def readNoiseSubset(splitType, noiseType):
    """
    Read <splitType> subset of noise. noiseType = IR or BG
    """
    if noiseType == 'BG':
        dirPath = os.path.join('noise_split', 'BG')
        
    if noiseType == 'IR':
        dirPath = os.path.join('noise_split', 'IR')
    
    splitPath = os.path.join(dirPath, f'{splitType}_filtered.txt')
    with open(splitPath) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def genFullSplitCSV(splitType, dfPath, nIR, nBG):
    """
    For given splitType, execute this to save a dataframe containing the augmentation information.
    Columns include gtzan 10s audioPath, IR path, BG path, filename and label.
    This df acts as an 
    """
    
    df = pd.DataFrame(columns = ['audioPath', 'IRPath', 'BGPath', 'fname', 'label'])
    trainSubset = readGtzanSubset(splitType)
            # Read BG train subset
    BGtrainSubset = readNoiseSubset(splitType, 'BG')
            # Read IR train subset
    IRtrainSubset = readNoiseSubset(splitType, 'IR')

    # Initialize df
    if not os.path.exists(dfPath):
        os.makedirs(dfPath)
    dfPathSS = os.path.join(dfPath, f'{splitType}_AugDatasetDesc.csv')

    # Create an array with 3 files per subSet audio
    for audioFile in trainSubset:
        for i in np.arange(3):
            fname = f'{audioFile}.{i}'
            subdir = audioFile.split('.')[0]
            IRPath = readNoisePaths(IRtrainSubset, nIR, 'IR', pathType='array')
            BGPath = readNoisePaths(BGtrainSubset, nBG, 'BG', pathType='array')
            path = os.path.join('..', 'datasets/gtzan10s', subdir, f'{fname}.wav')
            df = df.append({'audioPath' : path, 'IRPath' : IRPath, 'BGPath' : BGPath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    
    # Save df
    df.to_csv(dfPathSS, index = False)

def genDatasetDF(dfPath, nIR, nBG):
    """
    Function to generate csv with every row being a combination of 10s audio, BG and IR.

    Args:
        dfPath: path to save the dataframe
        nBG: Number of BGs to convolve with one audio
        nIR: Number of IRs to convolve with one audio

    Returns:
        Nothing. Saves the dataframe

    """ 
    splitTypes = ['train', 'test', 'val']
    # Create train dataset
    if not os.path.exists(dfPath):
        os.makedirs(dfPath)
    for splitType in splitTypes:
        genFullSplitCSV(splitType, dfPath, nIR, nBG)

def saveAudioDataset(x, augmentBGAudioTrimmed, sr, output_path, splitType, label, fname, bgname, irname):
    # fnames, labels
    # Save clean audio in cleanAudio folder
    savePath = os.path.join(output_path, 'cleanAudio', splitType, label)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    sf.write(os.path.join(savePath, f'{fname}.wav'), x[:10*sr], sr, subtype='PCM_24')
    # Trim augmented audio
        
    # save in augmentedAudioTrimmed folder
    savePath = os.path.join(output_path, 'augmentedAudioTrimmed', splitType, label)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    sf.write(os.path.join(savePath, f'{fname}_{bgname}_{irname}.wav'), augmentBGAudioTrimmed, sr, subtype='PCM_24')



def augmentAudioSubset(a, min_snr, max_snr, dfPath, splitType):
    """
    Augment audio function (to be used within main augmentAudio function) to generate augmented audio 
    with balanced augmentation for one subset.

    Args:
        a: blend factor of convolution
        min_snr: minimum snr for noise addition
        max_snr: maximum snr for noise addition
        dfPath: datasetPath
        splitType: train/ test/ val
        dstype: Dataset for which augmentation is needed
    """

    dfPathSS = os.path.join(dfPath, f'{splitType}_AugDatasetDesc.csv')
    output_path = os.path.join(dfPath, 'audio')
    df = pd.read_csv(dfPathSS)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    audioPaths = df['audioPath'].tolist()
    IRPaths = df['IRPath'].tolist()
    BGPaths = df['BGPath'].tolist()
    fnames = df['fname'].tolist()
    labels = df['label'].tolist()
    for i in tqdm(np.arange(len(audioPaths))):
        try:
        # Read audio
            x, sr = librosa.load(audioPaths[i])

            # Convolve with IR
            IRPaths[i] = IRPaths[i][2:-2]
            conv, irname = addIR(x, IRPaths[i], a)

            # Add BG noise
            BGPaths[i] = BGPaths[i][2:-2]
            augmentBGAudio, bgname = addBG(conv, sr, BGPaths[i], min_snr, max_snr)
            augmentBGAudioTrimmed = augmentBGAudio[:10*sr]
            saveAudioDataset(x, augmentBGAudioTrimmed, sr, output_path, splitType, labels[i], fnames[i], bgname, irname)

        except Exception as e:
            print (e)
            continue

def augmentAudioWrapper(dfPath, min_snr, max_snr, a):
    """
    augmentAudioWrapper. Augments all splitTypes on executing.
    """
    # Path of one Subset file
    splitTypes = ['train', 'test', 'val']
    for splitType in splitTypes:
        augmentAudioSubset(a, min_snr, max_snr, dfPath, splitType)

def mainAugment(dfPath, nBG, nIR, min_snr, max_snr, a):
    """
    Main function for augmentation. Execute this to run the augmentation
    genDatasetDF() saves a df whose each row has the audio and corresponding augmentations to be applied
    augmentAudio() applies those augmentations and saves the audio in output_path as per dstype required
    """
    print ('Generating dataset file...')
    genDatasetDF(dfPath, nIR, nBG)
    print ('Dataset file generated! Generating dataset now')
    augmentAudioWrapper(dfPath, min_snr, max_snr, a)
    print ('Successfully generated!')

config = parser.parse_args()
print(config)