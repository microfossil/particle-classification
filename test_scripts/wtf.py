from miso.training.model_params import default_params
from miso.training.model_trainer import train_image_classification_model
from miso.data.download import download_images

params = default_params()

params['type'] = 'resnet50_tl'
params['name'] = 'afafd'
params['description'] = None

params['input_source'] = r'E:\Data\martinrad\images_20191114_150220'
params['input_source'] = r'E:\Data\ForamA\raw'
params['input_source'] = r'C:\Users\rossm\Documents\Data\Foraminifera\ForamA\made_square\images_20190920_133842'

params['data_min_count'] = 40
params['data_split'] = 0.200000
params['data_map_others'] = False
params['save_dir'] = r'E:\Data\Testing'
params['save_dir'] = r'C:\Users\rossm\Documents\Data\Testing'

params['save_model'] = False #'frozen'
params['save_mislabeled'] = False
params['img_height'] = 224
params['img_width'] = 224
params['img_channels'] = 3
params['use_augmentation'] = True
params['use_class_weights'] = True
params['alr_epochs'] = 10
params['batch_size'] = 64

model, vector_model, data_source, result = train_image_classification_model(params)
