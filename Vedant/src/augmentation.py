import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import soundfile as sf
import os
import random
import scipy.signal as sp
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, ApplyImpulseResponse

random.seed(42)

import argparse
parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--a', type=float, default=0.5, help='blend for IR convolution')
parser.add_argument('--max_snr', type=int, default=10, help='Max SNR for BG noise')
parser.add_argument('--n_BG', type=int, default=10, help='Number of BG samples chosen')
parser.add_argument('--n_IR', type=int, default=10, help='Number of IR samples chosen')
parser.add_argument('--augmentation_type', type=str, default="both", help='Hop Size')

config = parser.parse_args()
a = config.a   # blend factor
min_snr = 1.0
max_snr = config.max_snr
aug_type = config.augmentation_type
n_BG = config.n_BG
n_IR = config.n_IR

audioPath = f'../../datasets/gtzan/Data/genres_original'
irPath = f'../../datasets/IR_MIT/'
bgPath = '../../datasets/TAU/audio/'

outputPath = f'../../datasets/augmentation/{aug_type}'

def blendedConv(x, h, a):
    # convolves audio (x) with IR (h) with a blend factor of 0<=a<=1
    dim = len(h)
    g = sp.unit_impulse(len(h))
    f = a*h + (1-a)*g
    blendConv = np.convolve(x, f)
    return blendConv
def addIR(x, IRnoisePath, a):
    irname = IRnoisePath.split('/')[-1].split('.')[0]
    print(IRnoisePath)
    # print (f'{irname}_{audioName}.wav')
    ir, fs = librosa.load(IRnoisePath, sr=22050)
    ##### Convolve audio sample with the IR
    conv = blendedConv(x, ir, a)
    return conv, irname
def augment_file(x,sr, noise_path, min_snr=1, max_snr=10, aug_type='both'):
    augment = Compose(
            [
                AddBackgroundNoise(
                sounds_path= noise_path,
                min_snr_in_db=min_snr,
                max_snr_in_db=max_snr,
                noise_rms="relative",
                p=1,
                lru_cache_size=2),
            ]
        )
    try:
        bgName = noise_path.split('/')[-1].split('.')[0]
    except:
        bgName = "None"
    try:
        irname = IRnoisePath.split('/')[-1].split('.')[0]
    except:
        irname= "None"
    

    if aug_type == 'both':
        print("both")
        audio = augment(samples=x, sample_rate=sr)
        
    if aug_type == 'ir':
        print("IR")
    if aug_type == "bg":
        print("BG")

def readPaths(path, n):
    # Read n BGs in an array
    paths = glob.glob(path + "*.wav")
    # Select n BG files randomly
    nPaths = random.sample(paths, n)
    return nPaths

def augment_data(path, output_path, n_IR=5, n_BG=10, min_snr=1, max_snr=10, aug_type='both'):
    df = pd.DataFrame(columns = ['audioPath', 'IRPath', 'BGPath', 'fname', 'label'])
    dirName, subdirList, _ = next(os.walk(audioPath))
    i = 0
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        # print(fileList)
        for filename in tqdm(fileList):
            path = os.path.join(dirName, subdir, filename)
            fname = filename.split('/')[-1][:-4]
            nBGPaths = readPaths(bgPath, n_BG)
            for noisePath in nBGPaths:
                nIRPaths = readPaths(irPath, n_IR)
                for IRnoisePath in nIRPaths:
                    df = df.append({'audioPath' : path, 'IRPath' : IRnoisePath, 'BGPath' : noisePath, 'fname' : fname, 'label' : subdir}, ignore_index = True)
    audioPaths = df['audioPath'].tolist()
    IRPaths = df['IRPath'].tolist()
    BGPaths = df['BGPath'].tolist()
    fnames = df['fname'].tolist()
    labels = df['label'].tolist()

    filename=''
    for i in tqdm(np.arange(len(audioPaths))):
        # Read audio
        if audioPaths[i] != filename:
            filename = audioPaths[i]
            x, sr = librosa.load(filename)
        audio, bgname, irname = augment_file(x, sr, BGPaths[i], min_snr, max_snr, aug_type)
        outputFileName = f'{fnames[i]}_{bgname}_{irname}.wav'
        sf.write(os.path.join(output_path,outputFileName), audio, sr, subtype='PCM_24') 
if __name__ == '__main__':
    print("Running Augmentation")