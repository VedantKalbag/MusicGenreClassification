# Filename  -  Description of contents
1. training.py - Most current training setup (Does not have cross validation implemented yet)
2. preprocess_audio_baseline.py - Used to load raw audio from the GTZAN dataset and divide it into blocks (given block and hop size) along with extracting log mel-spectrogram
3. preprocess_augmented.py - Used to divide a large numpy array containing the mel spectrograms and labels for every block in the dataset into train-test-val splits
4. melspec.py - Training setup for recreating benchmark results for CNN and Xception models