import argparse
parser = argparse.ArgumentParser()
# Model configuration.
parser.add_argument('--sr', type=int, default=22050, help='sample rate')
parser.add_argument('--blocksize', type=int, default=3, help='Block Size')
parser.add_argument('--hopsize', type=int, default=1, help='Hop Size')

config = parser.parse_args()
print(f"Processing the dataset with sample rate: {config.sr}")
print(f"Processing the dataset with block size: {config.blocksize}s")
print(f"Processing the dataset with hop size: {config.hopsize}s")

import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import math
import soundfile as sf
from tqdm import tqdm

import os
print(os.getcwd())


np.random.seed(42)
def block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs   
    t_mid = t + (0.5*blockSize/fs)
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t,t_mid)

def process_files(path, blocksize, hopsize, sr):
    dirName, subdirList, _ = next(os.walk(path))
    labels = []
    audio_files = []
    mel_specs = []
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        print(f"Processing {subdir} files")
        for filename in tqdm(fileList):
            # print(f"processing {filename}")
            try:
                x,sr = librosa.load(os.path.join(path, subdir, filename), sr=sr) # GTZAN is at 22050
                xb,ts,_ = block_audio(x, blocksize*sr, hopsize*sr, sr)
                for i in range(xb.shape[0]):
                    labels.append(subdir)
                    mel_specs.append(librosa.power_to_db(librosa.feature.melspectrogram(y=xb[i], sr=sr, n_mels=128, fmax=sr/2), ref=np.max))
                    audio_files.append(xb[i])
                    # sf.write(f'{filename}_start_{ts[i]}.wav')

            except Exception as e:
                print(e)
                continue

    df = pd.DataFrame()
    df['audio'] = audio_files
    df['mel_spectrogram'] = mel_specs
    df['label'] = labels
    out = df.to_numpy()
    os.mkdir(f'../working_data/{blocksize}s_block_{hopsize}s_hop_{sr}_sr')
    np.save(f'../working_data/{blocksize}s_block_{hopsize}s_hop_{sr}_sr/data.npy', out)

if __name__ == '__main__':
    PATH = '../../datasets/gtzan/Data/genres_original'
    process_files(path=PATH, blocksize=config.blocksize, hopsize=config.hopsize, sr=config.sr)