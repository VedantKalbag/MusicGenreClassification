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
from keras import layers

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='num epochs')
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
parser.add_argument('--AEFeatureSet', type=bool, default=False, help="True for generating corresponding clean feature set for autoencoders")
parser.add_argument('--cleanTrain', type=bool, default=False, help="Type of data for training: clean/augmented")
parser.add_argument('--cleanVal', type=bool, default=False, help="Type of data for training: clean/augmented")
parser.add_argument('--dstype', type=str, default='harmonic', help="Type of dataset: Harmonic/ Percussive")

config = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=config.gpu
print("Train on multiple gpus: ", bool(config.multigpu))
print("Training on GPU",os.environ['CUDA_VISIBLE_DEVICES'])

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

class AugmentedDataset(keras.utils.Sequence):
    def __init__(self, mode, batch_size, path, dstype, cleanTrain, cleanVal):
        # Choose dstype as harmonic/ percussive and cleanVal T/F for clean or aug validation set
        self.batch_size = batch_size
        self.mode = mode
        self.path = path
        self.cleanTrain = cleanTrain
        self.cleanVal = cleanVal
        self.dstype = dstype

        if dstype == 'harmonic':
            if self.mode == 'train':
                if self.cleanTrain == False:
                    self.data_path = os.path.join(path, 'featuresAugH', 'train')
                if self.cleanTrain == True:
                    self.data_path = os.path.join(path, 'featuresCleanH', 'train')
                print (self.data_path)
            if self.mode == 'test':
                if self.cleanVal == False:
                    self.data_path = os.path.join(path, 'featuresAugH', 'test')
                if self.cleanVal == True:
                    self.data_path = os.path.join(path, 'featuresCleanH', 'test')
            if self.mode == 'val':
                if self.cleanTrain == False:
                    self.data_path = os.path.join(path, 'featuresAugH', 'val')
                if self.cleanTrain == True:
                    self.data_path = os.path.join(path, 'featuresCleanH', 'val')

        if dstype == 'percussive':
            if self.mode == 'train':
                if self.cleanTrain == False:
                    self.data_path = os.path.join(path, 'featuresAugP', 'train')
                if self.cleanTrain == True:
                    self.data_path = os.path.join(path, 'featuresCleanP', 'train')
                print (self.data_path)
            if self.mode == 'test':
                if self.cleanVal == False:
                    self.data_path = os.path.join(path, 'featuresAugP', 'test')
                if self.cleanVal == True:
                    self.data_path = os.path.join(path, 'featuresCleanP', 'test')
            if self.mode == 'val':
                if self.cleanTrain == False:
                    self.data_path = os.path.join(path, 'featuresAugP', 'val')
                if self.cleanTrain == True:
                    self.data_path = os.path.join(path, 'featuresCleanP', 'val')
        
        _,_,self.filenames = next(os.walk(self.data_path))
        
    def __len__(self):
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self,idx):
        X = np.empty((self.batch_size, 128,130))
        y = np.empty(self.batch_size)
        batch = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        for i, ID in enumerate(batch):
            tmp = np.load(os.path.join(self.data_path, ID), allow_pickle=True)
            X[i] = np.abs(tmp[0])
            y[i] = tmp[1]

        return X,y #bat[:,0], bat[:,1]

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch %5 ==0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("/models/model_{}.hd5".format(epoch))

def plot_history(modelname, history):
    with open(f'models/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.json', 'w') as fp:
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

    plt.savefig(f'models/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{config.epochs}epochs.png')

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
    

def main(path, num_epochs, modelname, batchSize, blockLength, hopLength, dstype, cleanTrain, cleanVal):
    train_dataset = AugmentedDataset('train', batchSize, path, dstype, cleanTrain, cleanVal)
    test_dataset = AugmentedDataset('test', batchSize, path, dstype, cleanTrain, cleanVal)
    val_dataset = AugmentedDataset('val', batchSize, path, dstype, cleanTrain, cleanVal)

    print('Running ', modelname, ' model')
    if bool(config.multigpu):
        strategy = tf.distribute.MirroredStrategy(['GPU:1','GPU:2'])
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            # model = get_compiled_model()
            model = build_model(modelname, next(iter(train_dataset))[0][0].shape, 10)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy',   metrics=['acc'])
    else:
        model = build_model(modelname, next(iter(train_dataset))[0][0].shape, 10)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy',   metrics=['acc'])
    checkpoint_path = f'tmp/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs'
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
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, callbacks=[model_checkpoint_callback,saver])
        test_acc = model.evaluate(test_dataset, return_dict=True)
        history.history['test_acc'] = test_acc['acc']
        history.history['test_loss'] = test_acc['loss']
        plot_history(modelname, history)
        model.save(f'models/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs.h5')
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, saving model")
        model.save(f'models/model_10s_{config.dstype}_train{config.cleanTrain}_val{config.cleanVal}_{modelname}_a{config.a}_min{config.min_snr}_max{config.max_snr}_{config.blockLength}block_{config.hopLength}slide_{num_epochs}epochs.h5')
        sys.exit()
    #plot_model(model, show_shapes=True, to_file='model' + str(i) + '.png')
    K.clear_session()


if __name__ == '__main__':
    # featureName = 'melspec'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print (config.cleanTrain)
    print (config.cleanVal)
    print (config.dstype)
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    batchSize = config.batch
    path = os.path.join(config.datasetDir, config.datasetName, 'features')
    main(path, config.epochs, config.model, config.batch, config.blockLength, config.hopLength, config.dstype, config.cleanTrain, config.cleanVal)

# if __name__ == '__main__':
#     # featureName = 'melspec'
#     os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     batchSize = config.batch
#     datasetPath = os.path.join(config.datasetDir, config.datasetName)
#     AE = config.AEFeatureSet
#     path = os.path.join(datasetPath, 'features')
#     main(path, config.epochs, batchSize, config.blockLength, config.hopLength, datasetPath, AE)
