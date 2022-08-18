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
from cqt import *
from hpss import genHPSSDS
from gtzanSplit import melspecSplit
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
# parser.add_argument('--overwrite', type=str, default="n", help="Overwrite existing directory - y/n")
# parser.add_argument('--dstype', type=str, default="normal", help="Type of dataset to generate: ['melspec', 'hpss', 'cqt', 'all']")
# config = parser.parse_args()

# datasetPath = os.path.join(config.datasetDir, config.datasetName)

def genMelSpecDS(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype):
    
    splitTypes = ['train', 'test', 'val']

    for splitType in splitTypes:
        audioDir = os.path.join(audioPath, 'augmentedAudioTrimmed', splitType)
        dirName, subdirList, _ = next(os.walk(audioDir))
        # Create list of files in each subdir
        dfClean = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
        dfAugTrimmed = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
        # idx = 0
        for subdir in subdirList:
            _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
            print(f"Processing {subdir} files")
            for filename in tqdm(fileList):
                fname = filename.split('/')[-1]
                cleanfname = fname.split('_')[0] + '.wav'
                try:
                    # augmented trimmed
                    fpath = os.path.join(audioDir, subdir, filename)
                    x, sr = librosa.load(fpath)
                    # clean
                    fpathClean = os.path.join(audioPath, 'cleanAudio', splitType, subdir, cleanfname)
                    xclean, srclean = librosa.load(fpathClean)
                    blockSamples = int(blockLength*sr)
                    hopSamples = int(hopLength*sr)
                    pin = 0
                    pend = len(x)-blockSamples
                    while pin <= pend:
                        chunk = x[pin:pin+blockSamples]
                        melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
                        if melSpecChunk.shape[1] == int(blockLength/3*130):
                            dfAugTrimmed = dfAugTrimmed.append({'melspec' : melSpecChunk, 'label' : subdir, 'filename': filename, 'startSample' : pin}, ignore_index = True)
                        
                        chunkClean = xclean[pin:pin+blockSamples]
                        melSpecChunkClean = librosa.power_to_db(librosa.feature.melspectrogram(y=chunkClean, sr=srclean, n_mels=128, fmax=sr/2), ref=np.max)
                        if melSpecChunk.shape[1] == int(blockLength/3*130):
                            dfClean = dfClean.append({'melspec' :  melSpecChunkClean, 'label' : subdir, 'filename': cleanfname, 'startSample' : pin}, ignore_index = True)
                        pin += hopSamples
                except Exception as e:
                    print (e)
                    continue
        print (splitType, ' dataset: Completed!')
        print (dfClean.count())
        print (dfAugTrimmed.count())
        outClean = dfClean.to_numpy()
        outAugTrimmed = dfAugTrimmed.to_numpy()
        np.save(os.path.join(savePath, f'cleanAudio_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), outClean)
        np.save(os.path.join(savePath, f'augmentedAudioTrimmed_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), outAugTrimmed)


def genDataset(datasetPath, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype):
    if dstype == 'melspec':
        print ('generating melspec')
        genMelSpecDS(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype)
        # Add DS split for melspec
        melspecSplit(datasetPath, a, min_snr, max_snr, blockLength, hopLength)

    if dstype == 'cqt':
        genCQTDS(datasetPath, blockLength, hopLength, a, min_snr, max_snr)
    if dstype == 'hpss':
        # check if melspec exists
        if os.path.exists(datasetPath, 'features', 'featuresClean', 'train', '*.npy'):
            genHPSSDS(datasetPath, blockLength)
        else:
            genMelSpecDS(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype)
            melspecSplit(datasetPath, a, min_snr, max_snr, blockLength, hopLength)
            genHPSSDS(datasetPath, blockLength)
    if dstype == 'all':
        print ('generating melspec')
        genMelSpecDS(audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, dstype)
        print ('splitting melspec')
        melspecSplit(datasetPath, a, min_snr, max_snr, blockLength, hopLength)
        print ('generating cqt')
        genCQTDS(datasetPath, blockLength, hopLength, a, min_snr, max_snr)
        print ('generating hpss')
        genHPSSDS(datasetPath, blockLength)
