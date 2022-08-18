from concurrent.futures.process import _MAX_WINDOWS_WORKERS
from socket import NI_DGRAM
import numpy as np
import librosa
import os, glob
from util import *
from augmentation import *
from featuresExtract import *
from gtzanSplit import *

import warnings
warnings.filterwarnings("ignore")

# parser = argparse.ArgumentParser()
# parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
# parser.add_argument('--datasetName', type=str, default="snr10Full", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
# parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
# parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
# parser.add_argument('--a', type=float, default=0.5, help='IR blend factor')
# parser.add_argument('--min_snr', type=float, default=1.0, help='Min SNR for background noise')
# parser.add_argument('--max_snr', type=float, default=10.0, help='Max SNR for background noise')
# parser.add_argument('--nBG', type=int, default=1, help='Number of BG noise samples applied to each sample')
# parser.add_argument('--nIR', type=int, default=1, help='Number of IR noise samples applied to each sample')
# config = parser.parse_args()

# datasetDir = config.datasetDir
# datasetName = config.datasetName
# nBG = config.nBG
# nIR = config.nIR
# max_snr = config.max_snr
# min_snr = config.min_snr
# blockLength = config.blockLength
# hopLength = config.hopLength
# a = config.a

# datasetPath = os.path.join(config.datasetDir, config.datasetName)

# Define directory and paths

# datasetName = 'snr5Full'
# datasetPath = os.path.join(datasetDir, datasetName)
# featuresPath = os.path.join(datasetPath, 'features')
# cleanFeaturesPath = os.path.join(featuresPath, 'featuresClean', 'train')
# augmentedFeaturesPath = os.path.join(featuresPath, 'featuresAugmentedTrimmed', 'train')

def getFeature(path):
    df = pd.DataFrame(np.load(path, allow_pickle=True), columns=['melspec'])
    # print (df.head())
    feature = np.stack(df[['melspec']].values)
    # print (feature[0][0])
    y = feature[1]
    return feature[0][0], y

def hpssDataset(featurePath, splitType, blockLength, clean):
    """
    Generate HPSS from individual .npy files
    faudioType == audio file type of melspec. featuresClean, featuresCleanPadded, featuresAugmented, featuresAugmentedTrimmed
    splitType = train, test, val

    """
    numFiles = len(glob.glob(os.path.join(featurePath, '*.npy')))

    for i in np.arange(numFiles):
        # print (i, splitType, clean)
        # Read corresponding npy file
        X, y = np.abs(getFeature(os.path.join(featurePath, f'{i}.npy')))
        # Extract hpss in H and P
        mask_H, mask_P = librosa.decompose.hpss(X, kernel_size=31, power=2.0, mask=False, margin=1.0)
        # Store H in H as per the same dir structure
        
        if clean == True:
            savePathH = os.path.join(featurePath, '..', '..', 'featuresCleanH', splitType)
            savePathP = os.path.join(featurePath, '..', '..', 'featuresCleanP', splitType)
            savePathH = os.path.normpath(savePathH)
            savePathP = os.path.normpath(savePathP)
            # print (savePathP)
            if not os.path.exists(savePathH):
                os.makedirs(savePathH)
            if not os.path.exists(savePathP):
                os.makedirs(savePathP)

            df = pd.DataFrame(columns = ['melspec', 'label'])
            df = df.append({'melspec' : mask_H, 'label' : y[0]}, ignore_index = True)
            for idx, row in tqdm(df.iterrows()):
                if X.shape[1] == 130:
                    np.save(os.path.join(savePathH, f'{i}.npy'),row[['melspec','label']])
            
            df2 = pd.DataFrame(columns = ['melspec', 'label'])
            df2 = df2.append({'melspec' : mask_P, 'label' : y[0]}, ignore_index = True)
            for idx, row in tqdm(df2.iterrows()):
                if X.shape[1] == 130:
                    np.save(os.path.join(savePathP, f'{i}.npy'),row[['melspec','label']])

            
        if clean == False:
            savePathH = os.path.join(featurePath, '..', '..', 'featuresAugH', splitType)
            savePathP = os.path.join(featurePath, '..', '..', 'featuresAugP', splitType)
            savePathH = os.path.normpath(savePathH)
            savePathP = os.path.normpath(savePathP)
            # print (savePathP)
            
            if not os.path.exists(savePathH):
                os.makedirs(savePathH)
            if not os.path.exists(savePathP):
                os.makedirs(savePathP)

            df = pd.DataFrame(columns = ['melspec', 'label'])
            df = df.append({'melspec' : mask_H, 'label' : y[0]}, ignore_index = True)
            for idx, row in tqdm(df.iterrows()):
                if X.shape[1] == int(blockLength/3*130):
                    np.save(os.path.join(savePathH, f'{i}.npy'),row[['melspec','label']])
            
            df2 = pd.DataFrame(columns = ['melspec', 'label'])
            df2 = df2.append({'melspec' : mask_P, 'label' : y[0]}, ignore_index = True)
            for idx, row in tqdm(df2.iterrows()):
                if X.shape[1] == int(blockLength/3*130):
                    np.save(os.path.join(savePathP, f'{i}.npy'),row[['melspec','label']])


def genHPSSDS(datasetPath, blockLength):
    splitTypes = ['train', 'test', 'val']
    for splitType in splitTypes:
        print (splitType)
        featureSetsPath = os.path.join(datasetPath, 'features')
        featurePathClean = os.path.join(featureSetsPath, 'featuresClean', splitType)
        featurePathAug = os.path.join(featureSetsPath, 'featuresAugmentedTrimmed', splitType)
        hpssDataset(featurePathClean, splitType, blockLength, True)
        hpssDataset(featurePathAug, splitType, blockLength, False)