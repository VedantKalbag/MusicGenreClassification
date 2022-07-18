import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, models, layers, mixed_precision
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, Conv1DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
from keras import backend as K
# from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import json
import sys, os, pickle, numpy as np, pandas as pd

np.random.seed(42)
# from kapre import STFT, Magnitude, MagnitudeToDecibel
# from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer


import json
from collections import namedtuple
from json import JSONEncoder

def customJSONDecoder(JSONDict):
    return namedtuple('Namespace', JSONDict.keys())(*JSONDict.values())

config = json.load(open('./autoencoder_config.json'), object_hook=customJSONDecoder)
print(config)

os.environ['CUDA_VISIBLE_DEVICES']=str(config.gpu)


class AudioLoader(keras.utils.Sequence):
    def __init__(self, mode, batch_size):
        self.batch_size = batch_size
        self.mode = mode
        self.pos = np.random.randint(0,220500-66150)
        np.random.seed(42)

        self.clean_suffix = 'clean'
        self.augmented_suffix = 'augmented'


        if self.mode == 'train': 
            self.noisy_path = os.path.join(config.data_path, self.augmented_suffix,'train')
            self.clean_path = os.path.join(config.data_path, self.clean_suffix,'train')
        if self.mode == 'test':
            self.noisy_path = os.path.join(config.data_path, self.augmented_suffix,'test')
            self.clean_path = os.path.join(config.data_path, self.clean_suffix,'test')
        if self.mode == 'val':
            self.noisy_path = os.path.join(config.data_path, self.augmented_suffix,'val')
            self.clean_path = os.path.join(config.data_path, self.clean_suffix,'val')
        # print(os.path.exists(self.data_path))
        _,_,self.filenames = next(os.walk(self.noisy_path))

    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        try:
            source = np.empty((self.batch_size, 66150))#220500))
            target = np.empty((self.batch_size, 66150))#220500))
            batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
            for i, ID in enumerate(batch):
                tmp = np.load(os.path.join(self.noisy_path, ID), allow_pickle=True)
                source[i] = tmp[self.pos:self.pos+66150]
                tmp2 = np.load(os.path.join(self.clean_path, f'{ID.split("_")[0]}.npy'), allow_pickle=True)
                target[i] = tmp2[self.pos:self.pos+66150]
            return source,target#X,y #bat[:,0], bat[:,1]
        except Exception as e:
            print(i, ID)
            print(e)

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 ==0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("../models/model_{}.hd5".format(epoch))

print("Creating data loaders")
train = AudioLoader('train', config.batch)
val = AudioLoader('val',config.batch)
test = AudioLoader('test', config.batch)


def get_model(m):
    if m == 'a':
        input_sig = Input(batch_shape=(config.batch,66150,1))
        x = Conv1D(32,2048, activation='relu', padding='same')(input_sig)
        d3 = Conv1DTranspose(32,2048,activation='relu', padding='same')(x)
        decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d3)
        model= Model(input_sig, decoded)
    if m == 'b':
        input_sig = Input(batch_shape=(config.batch,66150,1))
        # ENCODER
        x1 = Conv1D(128,2048, activation='relu', padding='same')(input_sig)
        x2 = Conv1D(64,1024, activation='relu',padding='same')(x1)
        x3 = Conv1D(32,512, activation='relu',padding='same')(x2)
        # BOTTLENECK
        B = Dense(24)(x3)
        # DECODER
        d1 = Conv1DTranspose(32,512, activation='relu',padding='same')(B)
        d2 = Conv1DTranspose(64, 1024, activation='relu',padding='same')(d1)
        d3 = Conv1DTranspose(128,2048,activation='relu', padding='same')(d2)
        # OUTPUT LAYER
        decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d3)
        model= Model(input_sig, decoded)
    if m == 'c':
        input_sig = Input(batch_shape=(config.batch,66150,1))
        x = Conv1D(32,512, activation='relu', padding='same')(input_sig)
        B = Dense(500)(x)
        d3 = Conv1DTranspose(32,512,activation='relu', padding='same')(B)
        decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d3)
        model= Model(input_sig, decoded)
    if m == 'd':
        input_sig = Input(batch_shape=(config.batch,66150,1))
        x = Conv1D(32,4096, activation='relu', padding='same')(input_sig)
        B = Dense(2048)(x)
        d3 = Conv1DTranspose(32,4096,activation='relu', padding='same')(B)
        decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d3)
        model= Model(input_sig, decoded)
    

    print(model.summary())
    return model

model = get_model(m=config.m)
# print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss=tf.keras.losses.MeanSquaredError())
print("Model compiled")

# MODEL TRAINING CALLBACKS
saver = CustomSaver()
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='max',
    min_delta=0.001,
    patience=10,
    verbose=1
)
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='training_log.csv',
    separator=',',
    append=False
)
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=0.00001
)
cb_backup = tf.keras.callbacks.BackupAndRestore(backup_dir="../models/tmp/backup")
cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir="../logs", histogram_freq=1)
print("Beginning training")
try:
    history = model.fit(train, validation_data=val, epochs=config.epochs, callbacks=[saver, cb_earlystop, cb_csvlogger, cb_reducelr, cb_backup, cb_tensorboard])

except KeyboardInterrupt:
    print("\nKeyboard Interrupt, saving model")
    model.save(f'../models/denoising_AE_incomplete')
    print(model.evaluate(test))
    sys.exit()
print(model.evaluate(test))

model.save(f'../models/denoising_AE')


plt.figure()
plt.plot(np.log10(history.history['loss']))
plt.plot(np.log10(history.history['val_loss']))
plt.savefig('autoencoder-training.pdf', bbox_inches='tight')

try:
    test_file = np.load('../../datasets/denoising/augmented/test/blues.00012.0_street_traffic-prague-1153-42960-a_h146_Outside_MITCampus_1txts.npy')
    input_tensor = np.expand_dims(test_file, axis=0)
    cleaned = np.squeeze(model(input_tensor).numpy())
    import soundfile as sf
    sf.write('denoised_file.wav', cleaned, 22050)
except:
    print("Failed running inference test")
    pass