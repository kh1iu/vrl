# Variational Relational Learning (VRL)

VRL is an unsupervised learning method for addressing the relational learning problem where we learn the underlying relationship between a pair of data irrespective of the nature of those data. This repository contains the code used to generate reproducible experimental results in the article:

* Kuang-Hung Liu (2022). Relational Learning with Variational Bayes. 


## Requirements

The code has been tested in Python 3.6 and Tensorflow 1.14.


## Overview

The repository is organized as follows:

* 'data' folder contains python scripts for generating paired training datasets.

* 'model' folder contains model definitions.

* 'lib' folder contains utility functions.

* 'train_vrl.py' implements VRL training.

* 'load_data.py' implements data loading and RPDA functions.

* 'validate.py' implements model validation including calculating unsupervised clustering accuracy.

## Usage

### Step 1: Download dataset and setup data path
Create directory for storing training dataset and update environment variable 'DATA_PATH'. For linux user, a helpful bash file 'config.sh' is provided that needs to be updated and
```
source config.sh
```
Download [MNIST](http://yann.lecun.com/exdb/mnist/), [Omniglot](https://github.com/brendenlake/omniglot/archive/refs/heads/master.zip), [Yale](https://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/YALE.tar.gz), [ExtendedYale](http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip), and [RAVDESS](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1) dataset to the data directory.

### Step 2: Building training dataset
run 'data/build_data.py' to generate paired training dataset.
```
python data/build_data.py
```

### Step 3: Train VRL model
```
python train_vrl.py [save_model_name] -e [0-9]
```
Trained model will be saved in the saved_model (auto created) directoryfor later use.

### Step 4: Validation
```
python validate.py [path_to_saved_mode] 
```

## License
This code is being shared under the [MIT license](LICENSE)