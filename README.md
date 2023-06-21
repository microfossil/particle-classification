# Overview

Training scripts for particle classification

# What's new



# Usage
TBC


# Server

```shell
flask miso.app.app.py
celery -A miso.app.app.celery_app worker --loglevel INFO -P threads -Q celery_gpu -n gpu_worker@%h
```


# Installation

## Install Anaconda

TBC

## Install TensorFlow 2.12.2

```shell
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Install this library

If installing from source, navigate inside this repo and run

```shell
pip install -e .
```

Otherwise, install from PyPi using

```shell
pip install miso
```
