"""
Test of training ResNet50 transfer learning network using dataset hosted on Google Drive
"""
from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model

params = default_params()

params['name'] = 'check'
params['description'] = None
params['type'] = 'resnet50_tl'
params['filters'] = 4
params['use_depthwise_conv'] = False

params['img_type'] = 'rgb'
params['img_height'] = 128
params['img_width'] = 128
params['img_channels'] = 3

params['batch_size'] = 64
params['max_epochs'] = 10000
params['alr_epochs'] = 40
params['alr_drops'] = 4
params['use_augmentation'] = False

params['input_source'] = r'/Users/mar76c/Development/data/SeagressValidated'
params['data_min_count'] = 40
params['use_class_weights'] = True
params['data_split'] = 0.2

params['output_dir'] = r'output/'
params['save_model'] = 'frozen'
params['save_mislabeled'] = False

params['delete_mmap_files'] = False
params['mmap_directory'] = r'/Users/mar76c/Development/data/'
params['use_mmap'] = True

model, vector_model, data_source, result = train_image_classification_model(params)
