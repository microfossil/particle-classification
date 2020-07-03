from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model
from miso.data.download import download_images

params = default_params()

params['type'] = 'resnet_nocyclic'
params['name'] = 'ResNetNoCyclic'
params['description'] = None
params['filters'] = 4
params['input_source'] = r'/media/mar76c/DATA/Data/Seagrass/project_patches.xml'
params['data_min_count'] = 40
params['data_split'] = 0.200000
params['data_map_others'] = False
params['use_mmap'] = False
params['output_dir'] = r'/media/mar76c/DATA/Data/Seagrass/TrainedModels'
params['save_model'] = 'frozen'
params['save_mislabeled'] = False
params['img_height'] = 128
params['img_width'] = 128
params['img_channels'] = 3
params['use_augmentation'] = True
params['use_class_weights'] = True
params['alr_epochs'] = 10
params['batch_size'] = 64
params['max_epochs'] = 50

model, vector_model, data_source, result = train_image_classification_model(params)
