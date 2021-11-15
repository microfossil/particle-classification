"""
Training parameters
"""
import json
import re
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
    use_asoftmax = False


class TrainingParameters(Parameters):
    batch_size = 64
    max_epochs = 10000
    alr_epochs = 10
    alr_drops = 4
    monitor_val_loss = False
    use_class_weights = True
    use_class_undersampling = False
    use_augmentation = True


class DatasetParameters(Parameters):
    num_classes = None
    source = None
    min_count = 10
    val_split = 0.2
    map_others = False
    random_seed = 0
    memmap_directory = None


class AugmentationParameters(Parameters):
    rotation = [0, 360]
    gain = [0.8, 1, 1.2]
    gamma = [0.5, 1, 2]
    bias = None
    zoom = [0.9, 1, 1.1]
    gaussian_noise = None
    random_crop = None
    orig_img_shape = [256, 256, 3]


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
            self.name = self.dataset.source + "_" + self.cnn.id
            self.name = re.sub('[^A-Za-z0-9]+', '-', self.name)
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
                    self.augmentation.orig_img_shape[2] = 3
                else:
                    shape[2] = 1
                    self.augmentation.orig_img_shape[2] = 3
            self.cnn.img_shape = shape
        elif self.cnn.id.endswith("_tl"):
            self.cnn.img_shape[2] = 3
        if self.name == "":
            self.name = self.dataset.source.replace("http://", "").replace("https://", "").replace("/", "-").replace("\\", "-") + "_" + self.cnn.id + "_" + self.cnn.img_shape + "_" + self.cnn.img_type


def get_default_shape(cnn_type):
    if cnn_type.endswith("_tl"):
        return TRANSFER_LEARNING_PARAMS[cnn_type.split('_')[0]].default_input_shape
    else:
        return [224, 224, None]


if __name__ == "__main__":
    m = MisoParameters()
    print(m.asdict())
    print(m.to_json())







