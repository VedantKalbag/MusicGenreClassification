import numpy as np
import pandas as pd
import os
import shutil
import glob
import librosa
import matplotlib.pyplot as plt
import math
import soundfile as sf
from tqdm import tqdm
import os
import scipy.signal as sp

def genDataset(path, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr):
    # path = os.path.join(path, audioPath)
    print (path)
    dirName, subdirList, _ = next(os.walk(path))
    # Create list of files in each subdir
    df = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        print(f"Processing {subdir} files")
        for filename in tqdm(fileList):
            fname = filename.split('/')[-1]
            try:
                fpath = f'{path}{subdir}/{filename}'
                x, sr = librosa.load(fpath)
                blockSamples = int(blockLength*sr)
                hopSamples = int(hopLength*sr)
                pin = 0
                pend = len(x)-blockSamples
                while pin <= pend:
                    chunk = x[pin:pin+blockSamples]
                    melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
                    df = df.append({'melspec' : melSpecChunk, 'label' : subdir, 'filename': fname}, ignore_index = True)
                    pin += hopSamples
            except Exception as e:
                print (e)
                continue
    print (df.count())
    out = df.to_numpy()
    np.save(f'{savePath}a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy', out)

def extractFeaturesFromFile(x, sr, blockLength, hopLength, df, fname, subdir):
    blockSamples = int(blockLength*sr)
    hopSamples = int(hopLength*sr)
    pin = 0
    pend = len(x)-blockSamples
    # print (f'pend: {pend}')
    while pin <= pend:
        chunk = x[pin:pin+blockSamples]
        melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
        df = df.append({'melspec' : melSpecChunk, 'label' : subdir, 'filename': fname}, ignore_index = True)
        pin += hopSamples

def delFilesInFolder(folderPath):
    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def genIRDataset(path, blockLength, hopLength, savePath):
    df = pd.DataFrame(columns = ['audio', 'melspec', 'label', 'filename'])
    for file in glob.glob(path):
        fname = file.split('/')[-1]
        print (fname)
        try:
            x, sr = librosa.load(file)
            print (sr)
            blockSamples = int(blockLength*sr)
            x = x[:blockSamples*10]
            hopSamples = int(hopLength*sr)
            pin = 0
            pend = len(x)-blockSamples
            while pin <= pend:
                chunk = x[pin:pin+blockSamples]
                melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
                df = df.append({'audio' : chunk, 'melspec' : melSpecChunk, 'label' : 'blues', 'filename': fname}, ignore_index = True)
                pin += hopSamples
        except Exception as e:
            print (e)
            continue
    
    print (df.head())
    out = df.to_numpy()
    np.save(f'{savePath}{blockLength}s_block_{hopLength}s_hop.npy', out)

def readFeatureFile(path, blockLength, hopLength):
    df = pd.DataFrame(np.load(path,allow_pickle=True),columns=['audio', 'melspec', 'label', 'filename'])
    print (df.head())
    print (df.count())
    return df

def TAUSummary(path): 
    _, _, fileList = next(os.walk(path))
    numFiles = len(fileList)
    df = pd.DataFrame()
    df['filename'] = fileList
    df[['bg', 'city', 'val1', 'val2', 'version']] = df['filename'].str.split('-', expand=True)
    summ = df.groupby(['bg', 'city']).count()
    print (df.nunique())
    print (summ)
    df.to_csv('TAUDesc.csv', index = False)
    summ.to_csv('TAUSumm.csv')
    return df

def blendedConv(x, h, a):
    # convolves audio (x) with IR (h) with a blend factor of 0<=a<=1
    dim = len(h)
    g = sp.unit_impulse(len(h))
    f = a*h + (1-a)*g
    blendConv = np.convolve(x, f)
    return blendConv

def IRReviewFileGen(path):
    filenames = np.array([])
    index = np.array([])
    envt = np.array([])
    rem = np.array([])
    for file in glob.glob(path):
        filenames = np.append(filenames, file)
        tmp = file.split('/')[-1].split('.')[0].split('_')
        strlength = len(tmp)
        index = np.append(index, tmp[0])
        rem = np.append(rem, tmp[-2])
        if strlength == 4:
            envt = np.append(envt, tmp[1])
        if strlength == 5:
            envt = np.append(envt, f'{tmp[1]}-{tmp[2]}')
        # print (tmp)
        # print (index[-1], rem[-1], envt[-1])

    df = pd.DataFrame()
    df['index'] = index
    df['envt'] = envt
    df['rem'] = rem
    df['fname'] = filenames
    print (df.head())
    df.to_csv('IRReview.csv', index = False)