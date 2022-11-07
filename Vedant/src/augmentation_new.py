TEST= True


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

def addIR(x, IRPath, a):
    irname = IRPath.split('/')[-1].split('.')[0]
    print(IRPath)
    # print (f'{irname}_{audioName}.wav')
    ir, fs = librosa.load(IRPath, sr=22050)
    ##### Convolve audio sample with the IR
    conv = blendedConv(x, ir, a)
    return conv, irname

def addNoise(x,sr, noise_path, max_snr):
    augment = Compose(
            [
                AddBackgroundNoise(
                sounds_path= noise_path,
                min_snr_in_db=1,
                max_snr_in_db=max_snr,
                noise_rms="relative",
                p=1,
                lru_cache_size=2),
            ]
        )
    noisy_signal = augment(samples=x, sample_rate=sr)

def get_splits():
    gtzan_train = pd.read_csv('../../Saksham/gtzan_split/train_filtered.txt',header=None).to_numpy()
    gtzan_test = pd.read_csv('../../Saksham/gtzan_split/test_filtered.txt',header=None).to_numpy()
    gtzan_val = pd.read_csv('../../Saksham/gtzan_split/val_filtered.txt',header=None).to_numpy()

    bg_train = pd.read_csv('../../Saksham/noise_split/BG/train_filtered.txt',header=None).to_numpy()
    bg_test = pd.read_csv('../../Saksham/noise_split/BG/test_filtered.txt',header=None).to_numpy()
    bg_val = pd.read_csv('../../Saksham/noise_split/BG/val_filtered.txt',header=None).to_numpy()
        
    ir_train = pd.read_csv('../../Saksham/noise_split/IR/train_filtered.txt',header=None).to_numpy()
    ir_test = pd.read_csv('../../Saksham/noise_split/IR/test_filtered.txt',header=None).to_numpy()
    ir_val = pd.read_csv('../../Saksham/noise_split/IR/val_filtered.txt',header=None).to_numpy()

    gtzan={}
    gtzan['train']=gtzan_train
    gtzan['test'] = gtzan_test
    gtzan['val'] = gtzan_val

    bg={}
    bg['train']=bg_train
    bg['test'] = bg_test
    bg['val'] = bg_val

    ir={}
    ir['train']=ir_train
    ir['test'] = ir_test
    ir['val'] = ir_val
    return gtzan, bg, ir

def augmentSingleFile(audio_path, bg_path, ir_path, max_snr, a):

    x,sr = librosa.load(audio_path)
    
    conv_audio, irname = addIR(x,ir_path,a)

    augmented_audio, bgname = addNoise(conv_audio,sr,bg_path, max_snr)

    # Crop the outputs to create the different output files
    x_len = x.shape[0]
    augmented_audio_cropped = augmented_audio[:x_len]

    assert augmented_audio_cropped.shape[0] == x.shape[0]
    return x, augmented_audio_cropped, sr

def ToolBlockAudio(x, iBlockLength, iHopLength, f_s):

    iNumBlocks = np.ceil(x.shape[0] / iHopLength).astype(int)
    # pad with block length zeros just to make sure it runs for weird inputs, too
    afAudioPadded = np.concatenate((x, np.zeros([iBlockLength+iHopLength, ])), axis=0)

    return np.vstack([np.array(afAudioPadded[n*iHopLength:n*iHopLength+iBlockLength]) for n in range(iNumBlocks)])

def blockSingleFile(x, augmented_audio_cropped, sr, blocklength, hoplength):
    x_blocked = ToolBlockAudio(x,blocklength, hoplength, sr)
    augmented_blocked = ToolBlockAudio(augmented_audio_cropped, blocklength, hoplength, sr)

    return x_blocked, augmented_blocked

def processSingleFile(audio_path, bg_path, ir_path, max_snr, a, output_path, split):
    x, augmented_x, sr = augmentSingleFile(audio_path, bg_path, ir_path, max_snr, a)
    x_blocked, augmented_blocked = blockSingleFile(x, augmented_x, sr)
    
    print(x_blocked.shape, augmented_blocked.shape)












if __name__ == '__main__':

    if TEST:
        audio_path='../../datasets/gtzan10s/blues/blues.00000.0.wav'
        bg_path='../../datasets/TAU/audio/airport-barcelona-0-0-a.wav'
        ir_path = '../../datasets/IR_MIT/h001_Bedroom_65txts.wav'
        max_snr = 5
        a = 0.5
        output_path = ''
        split=''
        x,sr = librosa.load(audio_path)
        conv_audio, irname = addIR(x,ir_path, 0.5)
        augmented_audio, bgname = addNoise(conv_audio,sr,bg_path, max_snr)
        print(augmented_audio.shape)

        # processSingleFile(audio_path, bg_path, ir_path, max_snr, a, output_path, split)