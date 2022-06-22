from util import *
import argparse
parser = argparse.ArgumentParser()
# Model configuration.
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
parser.add_argument('--batch', type=int, default = 512, help='Batch Size')
parser.add_argument('--suffix',type=str, default='', help='Model Name Suffix')

config = parser.parse_args()
import sys, os, pickle, numpy as np, pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']=config.gpu
print("Train on multiple gpus: ", bool(config.multigpu))
print("Training on GPU",os.environ['CUDA_VISIBLE_DEVICES'])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, mixed_precision
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
import sys
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import json

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, mode, batch_size):
        self.batch_size = batch_size
        self.mode = mode
        if self.mode == 'train':
            self.data_path = '../datasets/gtzan10sAug/vedant/train'
        if self.mode == 'test':
            self.data_path = '../datasets/gtzan10sAug/vedant/test'
        if self.mode == 'val':
            self.data_path = '../datasets/gtzan10sAug/vedant/val'
        _,_,self.filenames = next(os.walk(self.data_path))
        
    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = tmp[0]
            y[i] = tmp[1]

        # bat = np.array([np.load(os.path.join(self.data_path, x), allow_pickle=True) for x in batch])
        return X,y #bat[:,0], bat[:,1]

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch %5 ==0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("/models/model_{}.hd5".format(epoch))

def plot_history(number, history):
    with open(f'model_10s_{config.audioType}_{modelname}_{config.audioType}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide.json', 'w') as fp:
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

    plt.savefig(f'models/model_10s_{config.audioType}_{modelname}_{config.audioType}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide.png')

def build_model(name, shape, output):
    if name == 'CNN':
        inputs = layers.Input(shape=shape)
        # layer 1
        x = layers.Reshape((shape[0], shape[1], 1))(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # layer 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        # last layer
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(output, activation='softmax', dtype='float32')(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    elif name == 'ResNet':
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)

        # layer 1
        first = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(first)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 2
        z = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        z = layers.MaxPooling2D(2, 1, padding='same')(z)
        # layer 3
        z = layers.Conv2D(256, 4, padding='same', activation='relu')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.BatchNormalization()(z)
        # Residual block
        z = layers.add([first, z])
        # Parallel pooling
        x = layers.MaxPooling2D(125, 1, padding='same')(z)
        y = layers.AveragePooling2D(125, 1, padding='same')(z)
        x = layers.Flatten()(x)
        y = layers.Flatten()(y)

        z = layers.add([x, y])
        # Dense layers
        z = layers.Dense(300, activation='relu')(z)
        z = layers.Dense(150, activation='relu')(z)
        outputs = layers.Dense(output, activation='softmax', dtype='float32')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    elif name == 'Xception':
        # EfficientNetB0
        inputs = keras.Input(shape=shape)

        z = layers.Reshape((shape[0], shape[1], 1))(inputs)
        eff = keras.applications.EfficientNetB0(weights=None, input_tensor=z, include_top=False)
        z = eff(z)
        z = layers.GlobalAveragePooling2D()(z)

        outputs = layers.Dense(output, activation='softmax', dtype='float32')(z)

        return keras.Model(inputs=inputs, outputs=outputs)
    

def main(path, num_epochs, modelname, batchSize, blockLength, hopLength):
    train_dataset = AugmentedDataset('train', batchSize)
    test_dataset = AugmentedDataset('test', batchSize)
    val_dataset = AugmentedDataset('val', batchSize)
    
    le = preprocessing.LabelEncoder()

    # TODO: LOAD X AND y
    df = readFeatureFile(path, blockLength, hopLength)
    X = np.stack(df['melspec'].values)
    y = df['label'].to_numpy()
    print (f'X shape: {X.shape}, Y shape: {y.shape}')
    # y = le.fit_transform(y)
    # Checkpoint: Clear

    # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)
    # x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.50, shuffle=True, random_state=42)
    # train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train = train.shuffle(buffer_size=200).batch(batchsize)
    # val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchsize)
    # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsize)
    # trans = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

   # train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   # train = train.shuffle(buffer_size=1000).batch(256)
   # val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(256)
   # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    print('Running ', modelname, ' model')
    if bool(config.multigpu):
        strategy = tf.distribute.MirroredStrategy(['GPU:1','GPU:2'])
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            # model = get_compiled_model()
            model = build_model(modelname, (x_train.shape[1], x_train.shape[2]), 10)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy',   metrics=['acc'])
    else:
        model = build_model(modelname, (x_train.shape[1], x_train.shape[2]), 10)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy',   metrics=['acc'])
    checkpoint_path = f'tmp/model_10s_{config.audioType}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # checkpoint_filepath = './tmp/checkpoint_{epoch}.ckpt'
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
    model.save_weights(checkpoint_path+"-{epoch}/model.ckpt".format(epoch=0))
    try:
        history = model.fit(train_dataset, validation_data=val, epochs=num_epochs, callbacks=[model_checkpoint_callback,saver])
        history = model.fit(X, y, validation_data=val, epochs=num_epochs, callbacks=[model_checkpoint_callback,saver])
        test_acc = model.evaluate(test, return_dict=True)
        history.history['test_acc'] = test_acc['acc']
        history.history['test_loss'] = test_acc['loss']
        plot_history(modelname, history)
        model.save(f'models/model_10s_{config.audioType}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide.h5')
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, saving model")
        model.save(f'models/model_10s_{config.audioType}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide.h5')
        sys.exit()
    #plot_model(model, show_shapes=True, to_file='model' + str(i) + '.png')
    K.clear_session()


if __name__ == '__main__':
    # featureName = 'melspec'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    path = f'features/10s/{config.blockLength}s_block_{config.hopLength}s_hop.npy'
    batchSize = config.batch
    
    main(path, config.epochs, config.model, config.batch, config.blockLength, config.hopLength)