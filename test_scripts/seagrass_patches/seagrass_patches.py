from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model
from miso.data.download import download_images

params = default_params()

params['type'] = 'resnet_nocyclic'
params['name'] = 'ResNetNoCyclic'
params['description'] = None
params['filters'] = 4
params['input_source'] = r'D:\Datasets\Seagrass\SeagrassFramesPatches_128_8_2-4-8'
params['input_source'] = r"D:\Datasets\Seagrass\project_seagrass_multiscale_128_8_2-4-8_10000.xml"
params['data_min_count'] = 40
params['data_split'] = 0.200000
params['data_map_others'] = False
params['use_mmap'] = True
params['mmap_directory'] = r"D:\Temp"
params['delete_mmap_files'] = False
params['output_dir'] = r'D:\Datasets\Seagrass\TrainedModels'
params['save_model'] = 'frozen'
params['save_mislabeled'] = False
params['img_height'] = 128
params['img_width'] = 128
params['img_channels'] = 3
params['use_augmentation'] = True
params['use_class_weights'] = True
params['alr_epochs'] = 40
params['batch_size'] = 8
params['max_epochs'] = 100

model, vector_model, data_source, result = train_image_classification_model(params)
