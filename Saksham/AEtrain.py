from util import *
import argparse
import sys, os, pickle, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import sys
from keras import backend as K
import matplotlib.pyplot as plt
import json
from autoencoder import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15, help='num epochs')
parser.add_argument('--model', type=str, default='Xception', help='model')
parser.add_argument('--multigpu', type=int, default=0, help='Train on multiple GPUs')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use for training')
parser.add_argument('--log_step', type=int, default=10, help='log interval')
parser.add_argument('--blockLength', type=float, default = 3.0, help='Analysis window size')
parser.add_argument('--hopLength', type=float, default = 0.5, help='Slide duration')
parser.add_argument('--audioType', type=str, default = 'Augmented', help='Slide duration')
parser.add_argument('--a', type=float, default = 0.5, help='Blend factor')
parser.add_argument('--min_snr', type=float, default = 1.0, help='Minimum snr for BG audio')
parser.add_argument('--max_snr', type=float, default = 10.0, help='Maximum snr for BG audio')
parser.add_argument('--batch', type=int, default = 256, help='Batch Size')
parser.add_argument('--datasetDir', type=str, default="../datasets/gtzan10sAug/datasets/", help="Main directory with subdirs of all generated datasets")
parser.add_argument('--datasetName', type=str, default="test", help="Output path of augmented audio subdirectories. Do not add a '/' after dir name in input")
parser.add_argument('--balAug', type=bool, default=True, help="Use balanced train test split for all augmentations")
parser.add_argument('--AEFeatureSet', type=bool, default=True, help="True for generating corresponding clean feature set for autoencoders")

config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu
print("Train on multiple gpus: ", bool(config.multigpu))
print("Training on GPU",os.environ['CUDA_VISIBLE_DEVICES'])

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def zeroPad(array, remFrames):
    bands = array.shape[0]
    pad = np.zeros(bands).reshape(bands, 1)
    for i in np.arange(remFrames):
        array = np.append(array, pad, axis=1)
    return array

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, mode, batch_size, path, datasetPath, AE):
        self.batch_size = batch_size
        self.mode = mode
        self.AE = AE
        self.datasetPath = datasetPath
        if self.mode == 'train':
            self.data_path = os.path.join(path, 'featuresAugmented', 'train')
            print (self.data_path)
            if AE == True:
                self.clean_data_path = os.path.join(path, 'featuresCleanPadded', 'train')
        if self.mode == 'test':
            self.data_path = os.path.join(path, 'featuresAugmented', 'test')
            print (self.data_path)
            if AE == True:
                self.clean_data_path = os.path.join(path, 'featuresCleanPadded', 'test')
        if self.mode == 'val':
            self.data_path = os.path.join(path, 'featuresAugmented', 'val')
            print (self.data_path)
            if AE == True:
                self.clean_data_path = os.path.join(path, 'featuresCleanPadded', 'val')
        
        _,_,self.filenames = next(os.walk(self.data_path))
        # if AE == True:
        #     _,_,self.filenamesClean = next(os.walk(self.clean_data_path))
        
    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        Xclean = np.empty((self.batch_size, 128,130))
        yclean = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        # if AE == True:
            # batchClean = self.filenamesClean[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = np.abs(tmp[0])
            y[i] = tmp[1]
            if AE == True:
                tmp = np.load(os.path.join(self.clean_data_path, ID), allow_pickle=True)
                Xclean[i] = np.abs(tmp[0])
                yclean[i] = tmp[1]

        # bat = np.array([np.load(os.path.join(self.data_path, x), allow_pickle=True) for x in batch])
        if AE == False:
            return X,y #bat[:,0], bat[:,1]
        if AE == True:
            return X, Xclean

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch %5 ==0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("/models/model_{}.hd5".format(epoch))

def loadData(path, num_epochs, batchSize, datasetPath, AE):
    train_dataset = AugmentedDataset('train', batchSize, path, datasetPath, AE)
    test_dataset = AugmentedDataset('test', batchSize, path, datasetPath, AE)
    val_dataset = AugmentedDataset('val', batchSize, path, datasetPath, AE)

    return train_dataset, test_dataset, val_dataset

def train(X_train, Xclean_train, learning_rate, batchSize, num_epochs):
    autoencoder = Autoencoder(
        input_shape=(128, 130, 1),
        conv_filters=(32, 64),
        conv_kernels=(3, 3),
        conv_strides=(1, 2),
        # conv_filters=(32, 64, 64, 64),
        # conv_kernels=(3, 3, 3, 3),
        # conv_strides=(1, 2, 2, 1),
        latent_space_dim=8
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(X_train, Xclean_train, batchSize, num_epochs)

def plot_history(history):
    with open(f'models/model_10s_abs_AE{config.AEFeatureSet}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.json', 'w') as fp:
        json.dump(history.history, fp, indent=4)

    fig, ax = plt.subplots(2, figsize=(10,8))

    # create accuracy subplot
    ax[0].plot(history.history['acc'], label='train accuracy')
    ax[0].plot(history.history['val_acc'], label='test accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_title('Accuracy eval')

    # create error subplot
    ax[1].plot(history.history['loss'], label='train error')
    ax[1].plot(history.history['val_loss'], label='test error')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Loss eval')

    plt.savefig(f'models/model_10s_abs_AE{config.AEFeatureSet}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.png')



def main(path, num_epochs, batchSize, blockLength, hopLength, datasetPath, AE):
    train_dataset, test_dataset, val_dataset = loadData(path, num_epochs, batchSize, datasetPath, AE)
    learning_rate = 0.0005
    
    print('Running model')
    
    autoencoder = Autoencoder(
        input_shape=(128, 130, 1),
        conv_filters=(32, 64),
        conv_kernels=(3, 3),
        conv_strides=(1, 2),
        # conv_filters=(32, 64, 64, 64),
        # conv_kernels=(3, 3, 3, 3),
        # conv_strides=(1, 2, 2, 1),
        latent_space_dim=8
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    
    if bool(config.multigpu):
        strategy = tf.distribute.MirroredStrategy(['GPU:1','GPU:2'])
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            # autoencoder = train(X_train, Xclean_train, learning_rate, batchSize, num_epochs)
            # history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, callbacks=[model_checkpoint_callback,saver])
            autoencoder.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',   metrics=['acc'])
            history = autoencoder.model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)
    else:
        # autoencoder = train(X_train, Xclean_train, learning_rate, batchSize, num_epochs)
        autoencoder.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',   metrics=['acc'])
        history = autoencoder.model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)
    checkpoint_path = f'tmp/model_10s_abs_AE{config.AEFeatureSet}_{config.audioType}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    saver = CustomSaver()
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False,
        save_freq = config.log_step*678
        )
    autoencoder.model.save_weights(checkpoint_path+"-{epoch}/model.ckpt".format(epoch=0))
    try:
        history = autoencoder.model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, callbacks=[model_checkpoint_callback,saver])
        test_acc = autoencoder.model.evaluate(test_dataset, return_dict=True)
        history.history['test_acc'] = test_acc['acc']
        history.history['test_loss'] = test_acc['loss']
        plot_history(modelname, history)
        model.save(f'models/model_10s_abs_AE{config.AE}_{config.audioType}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs.h5')
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, saving model")
        model.save(f'models/model_10s_abs_AE{config.AE}_{config.audioType}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs.h5')
        sys.exit()
    plot_model(model, show_shapes=True, to_file='model' + str(i) + '.png')
    K.clear_session()


if __name__ == '__main__':
    # featureName = 'melspec'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batchSize = config.batch
    datasetPath = os.path.join(config.datasetDir, config.datasetName)
    AE = config.AEFeatureSet
    path = os.path.join(datasetPath, 'features')
    main(path, config.epochs, batchSize, config.blockLength, config.hopLength, datasetPath, AE)
