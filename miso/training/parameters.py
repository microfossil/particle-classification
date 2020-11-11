"""
Training parameters
"""
import json
from collections import OrderedDict

from miso.models.transfer_learning import TRANSFER_LEARNING_PARAMS


class Parameters(object):
    def asdict(self):
        d = OrderedDict()
        for name in dir(self):
            v = getattr(self, name)
            if not name.startswith('__') and not callable(v):
                if isinstance(v, Parameters):
                    d[name] = v.asdict()
                else:
                    d[name] = v
        return d

    def to_json(self):
        d = self.asdict()
        return json.dumps(d)


class CNNParameters(Parameters):
    id = "base_cyclic"
    img_shape = [128, 128, 1]
    img_type = "greyscale"
    filters = 4
    blocks = None
    dense = None
    use_batch_norm = True
    global_pooling = None
    activation = "relu"
    use_msoftmax = False


class TrainingParameters(Parameters):
    batch_size = 64
    max_epochs = 10000
    alr_epochs = 10
    alr_drops = 4
    use_class_weights = True


class DatasetParameters(Parameters):
    num_classes = None
    source = None
    min_count = 10
    test_split = 0.2
    map_others = False
    random_seed = 0
    memmap_directory = None


class AugmentationParameters(Parameters):
    use_augmentation = True
    rotation = [0, 360]
    gain = [0.8, 1, 1.2]
    gamma = [0.5, 1, 2]
    bias = None
    zoom = [0.9, 1, 1.1]
    gaussian_noise = None
    random_crop = None


class OutputParameters(Parameters):
    output_dir = None
    save_model = True
    save_mislabeled = True


class MisoParameters(Parameters):
    name = ""
    description = ""
    cnn = CNNParameters()
    dataset = DatasetParameters()
    training = TrainingParameters()
    augmentation = AugmentationParameters()
    output = OutputParameters()

    def sanitise(self):
        if self.name == "":
            self.name = self.dataset.source.replace("http://","").replace("https://", "").replace("/","-").replace("\\","-") + "_" + self.cnn.id
        if self.cnn.img_shape is None:
            if self.cnn.id.endswith("_tl"):
                shape = TRANSFER_LEARNING_PARAMS[self.cnn.id.split('_')[0]].default_input_shape
            else:
                if self.cnn.id.startswith("base_cyclic") or self.cnn.id.startswith("resnet_cyclic"):
                    shape = [128, 128, 3]
                else:
                    shape = [224, 224, 3]
                if self.cnn.img_type == 'rgb':
                    shape[2] = 3
                else:
                    shape[2] = 1
            self.cnn.img_shape = shape
        if self.name == "":
            self.name = self.dataset.source.replace("http://", "").replace("https://", "").replace("/", "-").replace("\\", "-") + "_" + self.cnn.id + "_" + self.cnn.img_shape + "_" + self.cnn.img_type


if __name__ == "__main__":
    m = MisoParameters()
    print(m.asdict())
    print(m.to_json())

    # def sanitise(self):
    #
    #     # Make sure image shape is 3 for transfer learning
    #     if self.cnn_type.endswith("tl"):
    #         self.img_shape[2] = 3
    #     # Make sure name is somewhat descriptive
    #     if self.name == "":
    #         self.name = self.cnn_type


def get_default_shape(cnn_type):
    if cnn_type.endswith("_tl"):
        return TRANSFER_LEARNING_PARAMS[cnn_type.split('_')[0]].default_input_shape
    else:
        return [224, 224, None]


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



