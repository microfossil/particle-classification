"""
Class holding a dictionary of model parameters

The model parameters can be used with the ... script to automatically
create and train a network

The parameters:

Network:
- type

Datasource:
- data_source
- data_min_count
- data_load_to_memory

Input:
- img_height
- img_width
- img_channels

Topology:
- cnn_blocks
- cnn_dropout
- cnn_dense

Training:
- training_optimiser = Adam
- training_lr = 1e-3
- training_batch_size = 64
- training_max_epochs = 300
- training_alr_epochs = 40
- training_alr_drops = 4
- training_alr_factor = 0.5

Augmentation
- aug_method = "tensorflow" or "keras"
- aug_rotation
- aug_gain
- aug_zoom
- aug_gamma
- aug_bias
- aug_noise
"""
def default_params():
    params = dict()

    # Network description
    params['name'] = 'default'
    params['description'] = None

    # Type of network
    params['type'] = 'base_cyclic'

    # Custom parameters for the base_cyclic network
    # - number of filters in first block
    params['filters'] = 4
    # - use batch normalisation?
    params['use_batch_norm'] = True
    # - global pooling: None, 'avg', 'max'
    params['global_pooling'] = None
    # - activation: 'relu', 'elu', 'selu'
    params['activation'] = 'relu'

    # Input
    params['img_size'] = None
    params['img_height'] = 128
    params['img_width'] = 128
    params['img_channels'] = 1

    # Training
    params['batch_size'] = 64
    params['max_epochs'] = 5000
    params['alr_epochs'] = 40
    params['alr_drops'] = 4

    # Data
    params['input_source'] = None
    params['output_dir'] = None
    params['data_min_count'] = 40
    params['data_split'] = 0.25
    params['data_split_offset'] = 0
    params['data_map_others'] = False
    params['seed'] = None
    params['use_class_weights'] = True
    params['class_mapping'] = None
    params['delete_mmap_files'] = True
    params['mmap_directory'] = None
    params['use_mmap'] = False

    # What to save
    # save_model:
    # - None:          Don't save
    # - 'saved_model': Tensorflow saved model format (model and weights separately)
    # - 'frozen':      Frozen model
    params['save_model'] = 'frozen'
    params['save_mislabeled'] = False

    # Augmentation
    params['use_augmentation'] = True
    params['aug_rotation'] = True
    params['aug_gain'] = [0.8, 1, 1.2]
    params['aug_gamma'] = [0.5, 1, 2]
    params['aug_bias'] = None
    params['aug_zoom'] = [0.9, 1, 1.1]
    params['aug_gaussian_noise'] = None

    return params



