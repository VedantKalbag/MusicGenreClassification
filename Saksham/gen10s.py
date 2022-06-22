# Generate 10s audio chunks for each file in gtzan and save them by subdirectories

import librosa
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm

def gen10sClips(path, savePath, dur):
    # Split 30s GTZAN clips into 10s clips
    dirName, subdirList, _ = next(os.walk(path))
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        print(f"Processing {subdir} files")
        for filename in tqdm(fileList):
            fname = filename.split('.')
            try:
                x, sr = librosa.load(os.path.join(path, subdir, filename))
                for i in np.arange(3):
                    tmp = x[i*sr*dur:(i+1)*sr*dur]
                    sf.write(f'{savePath}{str(subdir)}/{fname[0]}.{fname[1]}.{i}.wav', tmp, sr)
                    print (f'{savePath}{str(subdir)}/{fname[0]}.{fname[1]}.{i}.wav - Done')
            except Exception as e:
                print (e)
                continue


def makedirs(path, savePath):
    dirName, subdirList, _ = next(os.walk(path))
    for subdir in subdirList:
        os.makedirs(f'{savePath}{subdir}')

if __name__ == '__main__':
    path = '../datasets/gtzan/Data/genres_original'
    savePath = '../datasets/gtzan10s/'
    dur = 10
    
    gen10sClips(path, savePath, dur)
    # makedirs(path, savePath)