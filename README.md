# Particle Classification
Python scripts for particle classification

Used by particle trieur to perform model training.

There are two main branches:

* **legacy** corresponding to the libraries used in the old miso2 environment and Particle Trieur 3.0.4 and below
* **master** latest version used in the miso environment and Particle Trieur 3.0.5 and above

## Installation

This library needs to be installed to perform training with Particle Trieur. It can also be used stand-alone via the command line interface (CLI)

### Initial

```
conda create -n miso python=3.9
conda activate miso
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10.1
pip install miso
```

### Updating

```
conda activate miso
pip install -U miso
```

## Command line interface (CLI)

### Train a model

TBD

### Classify a folder of images

Classifies a folder of images and saves the result in a CSV. This CSV can be imported into Particle Trieur. If the images are organised by sample into subfolders, will extract the sample as the subfolder name, else specify the sample name manually.

```
Usage: python -m miso classify-folder [OPTIONS]

  Classify images in a folder and output the results to a CSV file.

Options:
  -m, --model PATH                Path to the model information.  [required]
  -i, --input PATH                Path to the directory containing images.
                                  [required]
  -o, --output PATH               Path where the output CSV will be saved.
                                  [required]
  -b, --batch_size INTEGER        Batch size for processing images.  [default:
                                  32]
  -f, --in_samples / --no-in_samples
                                  Set this flag if images are stored in
                                  subfolders, using the subfolder names as
                                  sample labels.  [default: no-in_samples]
  -s, --sample TEXT               Default sample name if not using
                                  subdirectories.  [default: unknown]
  --unsure_threshold FLOAT        Threshold below which predictions are
                                  considered unsure.  [default: 0.0]
```
