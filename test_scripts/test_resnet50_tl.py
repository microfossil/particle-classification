"""
Test of training ResNet50 transfer learning network using dataset hosted on Google Drive
"""
from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model

params = default_params()

# A short name to describe the network.
# The name will be used to construct the output directory name as well
params['name'] = 'test_resnet50_tl'

# Longer description.
# The longer description will be saved in the xml description of the network
# Set to None to be auto-generated
params['description'] = None

# The type of network. Choices are
# - base_cyclic
# - resnet18
# - resnet34
# - resnet50
# - resnet50_tl     (resnet50 using transfer learning)
params['type'] = 'resnet50_tl'

# The input dimensions of the image.
# For transfer learning, the dimensions will automatically be set to the appropriate size for the network.
params['img_height'] = 224
params['img_width'] = 224
params['img_channels'] = 1

# Training
# Batch size is number of images presented per training iteration.
# Lower to 32 or 16 if getting out-of-memory errors
params['batch_size'] = 64
# Maximum epochs after which training is definitely stopped.
# Keep at a high number like 10000 as training will normally
# be stopped by the adaptive learning rate system
params['max_epochs'] = 10000
# Number of epochs and drops for the adaptive learning rate system. (ALR)
# ALR will monitor the last alr_epochs worth of epochs during training.
# If the loss is not decreasing, the learning rate will be dropped by half.
# After alr_drops times of drops, training is stopped.
params['alr_epochs'] = 40
params['alr_drops'] = 4
# Use augmentation (transfer learning automatically sets this to false)
params['use_augmentation'] = False

# Input data source
# Can be local directory or URL to zip file
params['input_source'] = r'https://1drv.ws/u/s!AiQM7sVIv7fah4MN2gWCXDWX_DT0OA?e=Eu3lZh'
# Minimum number of images per class for that class to be included.
# Classes with only a few images are not worth using in training.
params['data_min_count'] = 40
# Weight the classes by count
params['use_class_weights'] = True
# Fraction of images used for testing to calculate accuracy etc
params['data_split'] = 0.25

# Output
params['output_dir'] = r'output/'
params['save_model'] = True
params['save_mislabeled'] = True

model, data_source, result = train_image_classification_model(params)
