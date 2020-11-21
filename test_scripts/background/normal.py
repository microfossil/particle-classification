"""
Test of training ResNet50 transfer learning network using dataset hosted on Google Drive
"""
from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model

params = default_params()

params['name'] = 'check'
params['description'] = None
params['type'] = 'base_cyclic'
params['filters'] = 4
params['use_depthwise_conv'] = False

params['img_type'] = 'greyscaledm'
params['img_height'] = 64
params['img_width'] = 64
params['img_channels'] = 2

params['batch_size'] = 64
params['max_epochs'] = 10000
params['alr_epochs'] = 40
params['alr_drops'] = 4
params['use_augmentation'] = False

params['input_source'] = r'C:\Users\marchanr\OneDrive\Datasets\Pollen\pollen_all'
params['input_source'] = r'D:\Datasets\Foraminifera\images_20200302_075521'
params['data_min_count'] = 40
params['use_class_weights'] = True
params['data_split'] = 0.2

params['save_dir'] = r'output/'
params['save_model'] = 'frozen'
params['save_mislabeled'] = False

params['delete_mmap_files'] = False
params['mmap_directory'] = r'D:\Temp'
params['use_mmap'] = True

model, vector_model, data_source, result = train_image_classification_model(params)
