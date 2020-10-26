"""
Training parameters
"""
from collections import OrderedDict


class TrainingParameters(object):
    # Network identifier
    name = ""
    description = "description"
    type = "base_cyclic"

    # Network hyper-parameters
    filters = 4
    use_batch_norm = True
    global_pooling = None
    activation = "relu"

    # Network input
    img_shape = [128, 128, 1]
    img_type = "greyscale"

    # Network output
    num_classes = None

    # Training parameters
    batch_size = 64
    max_epochs = 10000
    alr_epochs = 10
    alr_drops = 4
    use_class_weights = True

    # Augmentation
    use_augmentation = True
    aug_rotation = True
    aug_gain = [0.8, 1, 1.2]
    aug_gamma = [0.5, 1, 2]
    aug_bias = None
    aug_zoom = [0.9, 1, 1.1]
    aug_gaussian_noise = None

    # Data
    source = None
    min_count = 10
    test_split = 0.2
    map_others = False
    random_seed = 0
    memmap_directory = None

    # Output
    output_dir = None
    # save_model:
    # - None:          Don't save
    # - 'saved_model': Tensorflow saved model format (model and weights separately)
    # - 'frozen':      Frozen model
    save_model = 'frozen'
    save_mislabeled = True

    def sanitise(self):
        # Make sure image shape is 3 for transfer learning
        if self.type.endswith("tl"):
            self.img_shape[2] = 3
        # Make sure name is somewhat descriptive
        if self.name == "":
            self.name = self.type

    def asdict(self):
        return OrderedDict((name, getattr(self, name)) for name in dir(self) if not name.startswith('__') and not callable(getattr(self, name)))


# def default_params():
#     params = dict()

    # # Network description
    # params['name'] = 'default'
    # params['description'] = None
    #
    # # Type of network
    # params['type'] = 'base_cyclic'
    #
    # # Custom parameters for the base_cyclic network
    # # - number of filters in first block
    # params['filters'] = 4
    # # - use batch normalisation?
    # params['use_batch_norm'] = True
    # # - global pooling: None, 'avg', 'max'
    # params['global_pooling'] = None
    # # - activation: 'relu', 'elu', 'selu'
    # params['activation'] = 'relu'

    # Input
    # params['img_size'] = None
    # params['img_height'] = 128
    # params['img_width'] = 128
    # params['img_channels'] = 1

    # # Training
    # params['batch_size'] = 64
    # params['max_epochs'] = 5000
    # params['alr_epochs'] = 40
    # params['alr_drops'] = 4
    #
    # # Data
    # params['input_source'] = None
    # params['output_dir'] = None
    # params['data_min_count'] = 40
    # params['data_split'] = 0.25
    # params['data_split_offset'] = 0
    # params['data_map_others'] = False
    # params['seed'] = None
    # params['use_class_weights'] = True
    # params['class_mapping'] = None
    # params['delete_mmap_files'] = True
    # params['mmap_directory'] = None
    # params['use_mmap'] = False
    #
    # # What to save
    # # save_model:
    # # - None:          Don't save
    # # - 'saved_model': Tensorflow saved model format (model and weights separately)
    # # - 'frozen':      Frozen model
    # params['save_model'] = 'frozen'
    # params['save_mislabeled'] = False
    #
    # # Augmentation
    # params['use_augmentation'] = True
    # params['aug_rotation'] = True
    # params['aug_gain'] = [0.8, 1, 1.2]
    # params['aug_gamma'] = [0.5, 1, 2]
    # params['aug_bias'] = None
    # params['aug_zoom'] = [0.9, 1, 1.1]
    # params['aug_gaussian_noise'] = None
    #
    # return params



