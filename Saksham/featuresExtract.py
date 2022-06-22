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

if __name__ == '__main__':
    blockLength = 3.0
    hopLength = 0.5
    fs = 22050
    a = 0.5
    min_snr = 1.0
    max_snr = 10.0
    path = f'../datasets/gtzan10sAug/Final/audio/a{a}_min{min_snr}_max{max_snr}/'
    # print (path)
    savePath = f'../datasets/gtzan10sAug/Final/features/a{a}_min{min_snr}_max{max_snr}_'
    genDataset(path, savePath, blockLength, hopLength)