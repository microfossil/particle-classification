# Particle Classification
Python scripts for particle classification

Used by particle trieur to perform model training.

## Installation

```
conda create -n miso python=3.9
conda activate miso
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10.1
pip install miso
```

## Updating

```
conda activate miso
pip install -U miso
```

## Command line interface (CLI)

### Train a model

TBD

### Classify a folder of images

Classifies a folder of images and saves the result in a CSV. This CSV can be imported into Particle Trieur. If the images are organised by sample into subfolders it will use the subfolder name as the sample name, else specify the sample name manually.

```
Usage: python -m miso classify-folder [OPTIONS]

  Classify images in a folder and output the results to a CSV file.

Options:
  -m, --model PATH              Path to the model information.  [required]
  -i, --input PATH              Path to the directory containing images.
                                [required]
  -o, --output PATH             Path where the output CSV will be saved.
                                [required]
  -b, --batch_size INTEGER      Batch size for processing images.  [default:
                                32]
  -s, --sample TEXT             Default sample name if not using
                                subdirectories.  [default: unknown]
  -u, --unsure_threshold FLOAT  Threshold below which predictions are
                                considered unsure.  [default: 0.0]
```

### Morphology of a folder of plankton images

Does morphology on a folder of plankton images and saves the result in a CSV. This CSV can be imported into Particle Trieur as parameters. If the images are organised by sample into subfolders it will use the subfolder name as the sample name, else specify the sample name manually.

The plankton model must first be downloaded from here: https://1drv.ms/f/s!AiQM7sVIv7fanuNzfcw2O8kAnDU26Q?e=NA5ztG

Make sure to put the `model.onnx` and the `model_info.xml` files in the same folder

```
Usage: python -m miso segment-folder [OPTIONS]

  Segment images in a folder and output the results.

Options:
  -m, --model PATH          Path to the model information.  [required]
  -i, --input PATH          Path to the folder containing images.  [required]
  -o, --output PATH         Path where the morphology csv will be saved.
                            [required]
  -b, --batch_size INTEGER  Batch size for processing images.  [default: 32]
  -s, --sample TEXT         Default sample name  [default: unknown]
  -t, --threshold FLOAT     Threshold for segmentation.  [default: 0.5]
  --save-contours           Whether to save contours or not.
```