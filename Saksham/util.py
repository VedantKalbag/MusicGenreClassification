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
import gc
import scipy.signal as sp
import random
from sklearn import preprocessing

def genMelspec(audioType, audioPath, splitType, savePath, a, min_snr, max_snr, blockLength, hopLength):
    audioDir = os.path.join(audioPath, audioType, splitType)
    dirName, subdirList, _ = next(os.walk(audioDir))
    df = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
    for subdir in subdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        print(f"Processing {audioType} {subdir} files")
        for filename in tqdm(fileList):
                fname = filename.split('/')[-1]
                if audioType[:3] == 'aug':
                    fname = fname.split('_')[0] + '.wav'
                try:
                    fpath = os.path.join(audioDir, subdir, filename)
                    print (filename)
                    x, sr = librosa.load(fpath)
                    blockSamples = int(blockLength*sr)
                    hopSamples = int(hopLength*sr)
                    pin = 0
                    pend = len(x)-blockSamples
                    while pin <= pend:
                        chunk = x[pin:pin+blockSamples]
                        melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
                        df = df.append({'melspec' : melSpecChunk, 'label' : subdir, 'filename': fname}, ignore_index = True)
                        # print ('appended')
                        pin += hopSamples
                except Exception as e:
                    print (e)
                    continue
    print (splitType, ' dataset: Completed!')
    print (df.count())
    out = df.to_numpy()
    np.save(os.path.join(savePath, f'{audioType}_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), out)

def genDataset(path, audioPath, savePath, blockLength, hopLength, a, min_snr, max_snr, balAug, AEFeatureSet):
    
    if balAug == True:
        splitTypes = ['train', 'test', 'val']
        for splitType in splitTypes:

            audioDir = os.path.join(audioPath, 'augmentedAudio', splitType)
            dirName, subdirList, _ = next(os.walk(audioDir))
            # Create list of files in each subdir
            df = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
            if AEFeatureSet == True:
                dfClean = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
                dfAugTrimmed = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
                dfCleanPadded = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
            # idx = 0
            for subdir in subdirList:
                _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
                print(f"Processing {subdir} files")
                for filename in tqdm(fileList):
                    fname = filename.split('/')[-1]
                    cleanfname = fname.split('_')[0] + '.wav'
                    try:
                        fpath = os.path.join(audioDir, subdir, filename)
                        x, sr = librosa.load(fpath)
                        if AEFeatureSet == True:
                            fpathClean = os.path.join(audioPath, 'cleanAudio', splitType, subdir, cleanfname)
                            xclean, srclean = librosa.load(fpathClean)
                            fpathCleanPadded = os.path.join(audioPath, 'cleanAudioPadded', splitType, subdir, cleanfname)
                            xcleanPadded, srcleanPadded = librosa.load(fpathCleanPadded)
                            fpathAugTrimmed = os.path.join(audioPath, 'augmentedAudioTrimmed', splitType, subdir, filename)
                            xAugTrimmed, srAugTrimmed = librosa.load(fpathAugTrimmed)
                        blockSamples = int(blockLength*sr)
                        hopSamples = int(hopLength*sr)
                        pin = 0
                        pend = len(x)-blockSamples
                        while pin <= pend:
                            chunk = x[pin:pin+blockSamples]
                            melSpecChunk = librosa.power_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=sr/2), ref=np.max)
                            df = df.append({'melspec' : melSpecChunk, 'label' : subdir, 'filename': fname}, ignore_index = True)
                            if AEFeatureSet == True:
                                chunkClean = xclean[pin:pin+blockSamples]
                                chunkCleanPadded = xcleanPadded[pin:pin+blockSamples]
                                chunkAugTrimmed = xAugTrimmed[pin:pin+blockSamples]
                                melSpecChunkClean = librosa.power_to_db(librosa.feature.melspectrogram(y=chunkClean, sr=srclean, n_mels=128, fmax=sr/2), ref=np.max)
                                melSpecChunkCleanPadded = librosa.power_to_db(librosa.feature.melspectrogram(y=chunkCleanPadded, sr=srcleanPadded, n_mels=128, fmax=sr/2), ref=np.max)
                                melSpecChunkAugTrimmed = librosa.power_to_db(librosa.feature.melspectrogram(y=chunkAugTrimmed, sr=srAugTrimmed, n_mels=128, fmax=sr/2), ref=np.max)
                                dfClean = dfClean.append({'melspec' : melSpecChunkClean, 'label' : subdir, 'filename': cleanfname}, ignore_index = True)
                                dfCleanPadded = dfCleanPadded.append({'melspec' : melSpecChunkCleanPadded, 'label' : subdir, 'filename': cleanfname}, ignore_index = True)
                                dfAugTrimmed = dfAugTrimmed.append({'melspec' : melSpecChunkAugTrimmed, 'label' : subdir, 'filename': filename}, ignore_index = True)
                            pin += hopSamples
                    except Exception as e:
                        print (e)
                        continue
            print (splitType, ' dataset: Completed!')
            print (df.count())
            out = df.to_numpy()
            outClean = dfClean.to_numpy()
            outCleanPadded = dfCleanPadded.to_numpy()
            outAugTrimmed = dfAugTrimmed.to_numpy()
            np.save(os.path.join(savePath, f'augmentedAudio_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), out)
            np.save(os.path.join(savePath, f'cleanAudio_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), outClean)
            np.save(os.path.join(savePath, f'cleanAudioPadded_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), outCleanPadded)
            np.save(os.path.join(savePath, f'augmentedAudioTrimmed_{splitType}_a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), outAugTrimmed)

    
    if balAug == False:
        # path = os.path.join(path, audioPath)
        dirName, subdirList, _ = next(os.walk(audioPath))
        # Create list of files in each subdir
        df = pd.DataFrame(columns = ['melspec', 'label', 'filename'])
        for subdir in subdirList:
            _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
            print(f"Processing {subdir} files")
            for filename in tqdm(fileList):
                fname = filename.split('/')[-1]
                try:
                    fpath = os.join(audioPath, subdir, filename)
                    # fpath = f'{audioPath}{subdir}/{filename}'
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
        np.save(os.path.join(savePath, f'a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy'), out)
        # np.save(f'{savePath}a{a}_min{min_snr}_max{max_snr}_{blockLength}s_block_{hopLength}s_hop.npy', out)

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

def readSplitFile(path):
    """ Read filenames from the artist biased removed master files for train/ test/ val subset."""
    # Input: path - path to master train test validate split files containing filenames
    # returns lines: Array of all the files in that subset
    with open(path) as f:
        lines = f.readlines()
        # lines = [line[:-1] for line in lines]
        lines = [line.split('/')[1] for line in lines]
        lines = [line.split('.')[0] + '.' + line.split('.')[1] for line in lines]
        return lines

def loadDF(path, readFeatures = 'y', readLabels = 'y', readFnames = 'y', pkl=True):
    # Load the npy file containing features and labels for all samples
    df = pd.DataFrame(np.load(path, allow_pickle=pkl), columns=['melspec','label','filename'])
    # X = np.stack(df[['melspec']].values)
    if readFeatures == 'y':
        X = np.stack(df[['melspec', 'filename']].values)
    elif readFeatures == 'n':
        X = np.array([])
    
    if readLabels == 'y':
        y = df['label'].to_numpy()
    elif readLabels == 'y':
        y = np.array([])

    if readFnames == 'y':
        fnames = df['filename'].to_numpy()
    elif readFnames == 'n':
        fnames = np.array([])
    else:
        print ('invalid input')
    del df
    gc.collect()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)
    return X, y, fnames

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