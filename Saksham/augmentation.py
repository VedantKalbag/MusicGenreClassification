import numpy as np
import pandas as pd
import os
import glob
import librosa
import soundfile as sf
import random
# from tqdm import tqdm
import audiomentations
from util import *
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, ApplyImpulseResponse

a = 0.5   # blend factor
min_snr = 1.0
max_snr = 10.0

noisePath = '../datasets/TAU/audio/airport-barcelona-0-0-a.wav'
# audioName = 'blues.00000.0'
# audioPath = f'../datasets/gtzan10s/blues/{audioName}.wav'
audioPath = f'../datasets/gtzan10s/'
irPath = '../datasets/IR_MIT/'
bgPath = '../datasets/TAU/audio/'
IRsavePath = f'../datasets/gtzan10sAug/IR/audio/a{a}/'
dfPath = '../datasets/gtzan10sAug/AugDatasetDesc.csv'
nBG = 5
nIR = 3

# Total num of audio files - 44946
# Total num of BG files - 14400

# Test
# blockLength = 3.0
# hopLength = 0.5

# audio, sr = librosa.load(audioPath) # Read one audio sample

# # Read all IRs
# def convIRs(x, sr, n, irPath, IRsavePath, audioPath, audioName, a):
#     # Convolve audioName with all IRs in a folder
#     allIRPaths = glob.glob(irPath + "*.wav")
#     # Select n files randomly
#     nIRPaths = random.sample(allBGPaths, n)

#     for file in glob.glob(nIRPaths):
#         irname = file.split('/')[-1].split('.')[0]
#         print (f'{irname}_{audioName}.wav')
#         ir, fs = librosa.load(file, sample_rate=22050)
#         ##### Convolve audio sample with the IR
#         conv = blendedConv(x, ir, a)
#         # Save audio
#         sf.write(f'{IRsavePath}{irname}_{audioName}.wav', conv, sr, subtype='PCM_24')

# def addBGs(x, sr, n, bgPath, BGsavePath, audioName, min_snr, max_snr):
#     # Add randomly selected 'n' BGs from the folder to 'audioName'. 
#     # bgPath: directory with all BG noises, BGsavePath: Folder to save audio with BG Noise.
#     # min and max snr in dB

#     allBGPaths = glob.glob(bgPath + "*.wav")
#     # Select n BG files randomly
#     nBGPaths = random.sample(allBGPaths, n)

#     for noisePath in nBGPaths:
#         bgName = noisePath.split('/')[-1].split('.')[0]
#         print (bgName)
#         augmentBG = Compose([
#         AddBackgroundNoise(
#             sounds_path= noisePath,
#             min_snr_in_db=min_snr,
#             max_snr_in_db=max_snr,
#             noise_rms="relative",
#             p=1,
#             lru_cache_size=2)
#         ])
#         augmentBGAudio = augmentBG(samples=x, sample_rate=sr)
#         sf.write(f'{BGsavePath}{audioName}_{bgName}.wav', augmentBGAudio, sr, subtype='PCM_24')

# BGsavePath = f'../datasets/gtzan10sAug/BG/audio/min{min_snr}_max{max_snr}/'

########## Final functions
# addBGs(audio, sr, n, bgPath, BGsavePath, audioName, min_snr, max_snr)
# path = '../datasets/gtzan10sAug/Final/audio/a0.5_min1.0_max10.0/pop/'
# delFilesInFolder(BGsavePath)

# Read nAudio audios from every subdir
# Read nBG BGs
# Read nIR IRs

def readNoisePaths(bgPath, n):
    # Read n BGs in an array
    allBGPaths = glob.glob(bgPath + "*.wav")
    # Select n BG files randomly
    nBGPaths = random.sample(allBGPaths, n)
    return nBGPaths

def addBG(x, sr, noisePath, min_snr, max_snr):
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
    irname = IRnoisePath.split('/')[-1].split('.')[0]
    print(IRnoisePath)
    # print (f'{irname}_{audioName}.wav')
    ir, fs = librosa.load(IRnoisePath, sr=22050)
    ##### Convolve audio sample with the IR
    conv = blendedConv(x, ir, a)
    return conv, irname

def genDatasetDF(audioPath, bgPath, irPath, nBG, nIR):
    df = pd.DataFrame(columns = ['audioPath', 'IRPath', 'BGPath', 'fname', 'label'])
    dirName, subdirList, _ = next(os.walk(audioPath))
    i = 0
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        # print(fileList)
        for filename in tqdm(fileList):
            path = os.path.join(dirName, subdir, filename)
            fname = filename.split('/')[-1][:-4]
            nBGPaths = readNoisePaths(bgPath, nBG)
            for noisePath in nBGPaths:
                nIRPaths = readNoisePaths(irPath, nIR)
                for IRnoisePath in nIRPaths:
                    df = df.append({'audioPath' : path, 'IRPath' : IRnoisePath, 'BGPath' : noisePath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    df.to_csv('../datasets/gtzan10sAug/AugDatasetDesc.csv', index = False)
    return df

def augmentAudio(dfPath, min_snr, max_snr, a):
    # read csv
    df = pd.read_csv(dfPath)
    print (df.head())
    audioPaths = df['audioPath'].tolist()
    IRPaths = df['IRPath'].tolist()
    BGPaths = df['BGPath'].tolist()
    fnames = df['fname'].tolist()
    labels = df['label'].tolist()
    for i in np.arange(len(audioPaths)):
        # Read audio
        x, sr = librosa.load(audioPaths[i])

        # Convolve with IR
        conv, irname = addIR(x, IRPaths[i], a)

        # Add BG noise
        augmentBGAudio, bgname = addBG(conv, sr, BGPaths[i], min_snr, max_snr)
        
        finalSavePath = f'../datasets/gtzan10sAug/Final/audio/a{a}_min{min_snr}_max{max_snr}/{labels[i]}/'
        finalFileName = f'{fnames[i]}_{bgname}_{irname}.wav'
        print(f'Done: {finalFileName}')
        sf.write(f'{finalSavePath}{finalFileName}', augmentBGAudio, sr, subtype='PCM_24')
        print (i)
        i += 1

augmentAudio(dfPath, min_snr, max_snr, a)

# df = genDatasetDF(audioPath, bgPath, irPath, nBG, nIR)