# Overview

The particle-classification repository provides training and inference scripts for particle classification.

The training methods and CNN models are based on the paper [Automated analysis of foraminifera fossil records by image classification using a convolutional neural network](https://jm.copernicus.org/articles/39/183/2020/) by Marchant et al., with some improvements.


# Installation

## Install Anaconda

Anaconda is needed to create a python environment to install and use this library.

Follow the [instructions on the Anaconda website](https://docs.anaconda.com/free/anaconda/install/) to install the anaconda package. Make sure to enter 'yes' when asked if you want to initialize Anaconda.

## Create environment

Once downloaded, open the terminal (Linux) or Anaconda Prompt (Windows). You should see `(base)` at the start of the terminal prompt which means you are in the default anaconda environment.

Create a new environment with python 3.11 as follows:
```shell
conda create -n miso python==3.11
```

## Install TensorFlow 2.12.2

Change into the new environment

```shell
conda activate miso
```

Install tensorflow 2.12

```shell
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Test if everything is working by running the following command

```shell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The result should be one (or more if you have them) GPUs, e.g.:

```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Install this library

Change into the new environment if not already

```shell
conda activate miso
```

If installing from source, navigate inside this repo and run

```shell
pip install -e .
```

Otherwise, install from PyPi using

```shell
pip install miso
```
