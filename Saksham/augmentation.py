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
parser.add_argument('--balAug', type=bool, default=True, help="Use balanced train test split for all augmentations")

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

def genFullSplitCSV(splitType, dfPath):
    
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
            IRPath = readNoisePaths(IRtrainSubset, 1, 'IR', pathType='array')
            BGPath = readNoisePaths(BGtrainSubset, 1, 'BG', pathType='array')
            path = os.path.join('..', 'datasets/gtzan10s', subdir, f'{fname}.wav')
            df = df.append({'audioPath' : path, 'IRPath' : IRPath, 'BGPath' : BGPath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    
    # Save df
    df.to_csv(f'{dfPathSS}', index = False)

def genDatasetDF(audioPath, dfPath, bgPath, irPath, nBG, nIR, balAug):
    """
    Function to generate csv with every row being a combination of 10s audio, BG and IR.
    audioPath: f'../datasets/gtzan10s/' # Rel. path of gtzan10s
    dfPath: path to save the final dataframe including save name. (Ends with *.csv)
    bgPath: Rel path of folder containing BGs
    irPath: Rel path of folder containing IRs
    nBG: Number of BGs to convolve with one audio
    nIR: Number of IRs to convolve with one audio
    balAugmentations: If True, split all augmentations like a pre-defined train and test.
    """
    print ('Val: ', dfPath)
    # Initialize df
    df = pd.DataFrame(columns = ['audioPath', 'IRPath', 'BGPath', 'fname', 'label'])
    
    if balAug == True:

        splitTypes = ['train', 'test', 'val']
        # Create train dataset
        for splitType in splitTypes:
            genFullSplitCSV(splitType, dfPath)
    
    if balAug == False:
    # Loop through all audio files
        dirName, subdirList, _ = next(os.walk(audioPath))
        i = 0
        for subdir in subdirList:
            _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
            print(subdir)
            
            # Read audio file names
            for filename in tqdm(fileList):
                path = os.path.join(dirName, subdir, filename)
                fname = filename.split('/')[-1][:-4]
                # Randomly select nBG noises
                nBGPaths = readNoisePaths(bgPath, nBG)
                for noisePath in nBGPaths:
                    # Randomly select nIRs
                    nIRPaths = readNoisePaths(irPath, nIR)
                    for IRnoisePath in nIRPaths:
                        # Append in df
                        df = df.append({'audioPath' : path, 'IRPath' : IRnoisePath, 'BGPath' : noisePath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    
        if not os.path.exists(dfPath):
            os.makedirs(dfPath)
        # Save df
        df.to_csv(f'{dfPath}', index = False)
    return df

def augmentAudioSubset(a, min_snr, max_snr, dfPath, splitType, AE):
    """
    Augment audio core function (to be used within main augmentAudio function) to generate augmented audio with balanced augmentation
    """
    dfPathSS = os.path.join(dfPath, f'{splitType}_AugDatasetDesc.csv')
    df = pd.read_csv(dfPathSS)
    output_path = os.path.join(dfPath, 'audio')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    audioPaths = df['audioPath'].tolist()
    IRPaths = df['IRPath'].tolist()
    BGPaths = df['BGPath'].tolist()
    fnames = df['fname'].tolist()
    labels = df['label'].tolist()
    for i in tqdm(np.arange(len(audioPaths))):
        print (i)
        try:
        # Read audio
            x, sr = librosa.load(audioPaths[i])

            # Convolve with IR
            IRPaths[i] = IRPaths[i][2:-2]
            conv, irname = addIR(x, IRPaths[i], a)

            # Add BG noise
            BGPaths[i] = BGPaths[i][2:-2]
            augmentBGAudio, bgname = addBG(conv, sr, BGPaths[i], min_snr, max_snr)

            savePath = os.path.join(output_path, 'augmentedAudio', splitType, labels[i])
            # print (savePath)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            sf.write(os.path.join(savePath, f'{fnames[i]}_{bgname}_{irname}.wav'), augmentBGAudio, sr, subtype='PCM_24')
            if AE == True:
                # print ('AE: True')
                # Save clean audio in cleanAudio folder
                savePath = os.path.join(output_path, 'cleanAudio', splitType, labels[i])
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                sf.write(os.path.join(savePath, f'{fnames[i]}.wav'), x, sr, subtype='PCM_24')
                # Trim augmented audio
                augmentBGAudioTrimmed = augmentBGAudio[:10*sr]    
                # save in augmentedAudioTrimmed folder
                savePath = os.path.join(output_path, 'augmentedAudioTrimmed', splitType, labels[i])
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                sf.write(os.path.join(savePath, f'{fnames[i]}_{bgname}_{irname}.wav'), augmentBGAudioTrimmed, sr, subtype='PCM_24')
                # Pad clean audio
                padLength = len(augmentBGAudio)-len(x)
                paddedCleanAudio = np.append(x, np.zeros(padLength))
                # Save in cleanAudioPadded folder
                savePath = os.path.join(output_path, 'cleanAudioPadded', splitType, labels[i])
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                sf.write(os.path.join(savePath, f'{fnames[i]}.wav'), paddedCleanAudio, sr, subtype='PCM_24')
        except Exception as e:
            print (e)
            continue

def augmentAudio(dfPath, output_path, min_snr, max_snr, a, balAug, AE):
    
    if balAug == True:
        # Path of one Subset file
        splitTypes = ['train', 'test', 'val']
        for splitType in splitTypes:
            augmentAudioSubset(a, min_snr, max_snr, dfPath, splitType, AE)
    
    if balAug == False:
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
            
            finalSavePath = os.path.join(output_path, 'audio', f'a{a}_min{min_snr}_max{max_snr}',labels[i])
            # finalSavePath = f'{output_path}/audio/a{a}_min{min_snr}_max{max_snr}/{labels[i]}/'
            if not os.path.exists(finalSavePath):
                os.makedirs(finalSavePath)
            finalFileName = f'{fnames[i]}_{bgname}_{irname}.wav'
            sf.write(os.path.join(finalSavePath, finalFileName), augmentBGAudio, sr, subtype='PCM_24')
            # sf.write(f'{finalSavePath}{finalFileName}', augmentBGAudio, sr, subtype='PCM_24')
            # i += 1

def mainAugment(audioPath, dfPath, output_path, bgPath, irPath, nBG, nIR, min_snr, max_snr, a, balAug, AE):
    print ('Generating dataset file...')
    df = genDatasetDF(audioPath, dfPath, bgPath, irPath, nBG, nIR, balAug)
    print ('Dataset file generated! Generating dataset now')
    augmentAudio(dfPath, output_path, min_snr, max_snr, a, balAug, AE)
    print ('Successfully generated!')

config = parser.parse_args()
print(config)