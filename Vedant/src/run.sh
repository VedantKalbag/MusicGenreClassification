#!/bin/zsh
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 1000 --v 1 --train_set "clean" 
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 1000 --v 1 --train_set "snr10" 
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 1000 --v 1 --train_set "snr5" 
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 1000 --v 1 --train_set "snr1" 

python3 jukebox_benchmarking.py --aggregate "mean" --num_features 4800 --v 1 --train_set "clean" --suffix "benchmark"
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 4800 --v 1 --train_set "snr10" --suffix "benchmark"
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 4800 --v 1 --train_set "snr5" --suffix "benchmark"
python3 jukebox_benchmarking.py --aggregate "mean" --num_features 4800 --v 1 --train_set "snr1" --suffix "benchmark"