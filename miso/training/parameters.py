"""
Training parameters
"""
from pprint import pprint
from typing import List, Optional
from dataclasses import dataclass, field
import re
from xml.dom.minidom import parseString

import dicttoxml
import xmltodict

from miso.models.keras_models import KERAS_MODEL_PARAMETERS
from miso.utils.base_config import BaseConfig, default_field


@dataclass
class ModelConfig(BaseConfig):
    # Common values
    type: str = "base_cyclic"
    img_shape: List[int] = field(default=(128, 128, 1))
    img_type: str = "greyscale"

    # Transfer learning cyclic layers
    use_tl_cyclic: bool = False
    use_tl_cyclic_gain: bool = False

    # Parameters for custom networks
    filters: Optional[int] = 4
    blocks: Optional[int] = None
    dense: Optional[int] = None
    use_batch_norm: bool = True
    global_pooling: bool = None
    activation: str = "relu"
    use_asoftmax: bool = False


@dataclass
class TrainingConfig(BaseConfig):
    batch_size: int = 64
    max_epochs: int = 10000
    alr_epochs: int = 10
    alr_drops: int = 4
    monitor_val_loss: bool = False
    use_class_weights: bool = True
    use_class_undersampling: bool = False
    use_augmentation: bool = True
    use_transfer_learning: bool = True
    transfer_learning_augmentation_factor: int = 0


@dataclass
class DatasetConfig(BaseConfig):
    num_classes: Optional[int] = None
    source: Optional[str] = None
    min_count: Optional[int] = 10
    train_split: Optional[float] = None
    val_split: Optional[float] = 0.2
    map_others: bool = False
    random_seed: int = 0
    memmap_directory: str = None


@dataclass
class AugmentationConfig(BaseConfig):
    rotation: List[float] = field(default=(0, 360))
    gain: Optional[List[float]] = field(default=(0.8, 1, 1.2))
    gamma: Optional[List[float]] = field(default=(0.5, 1, 2))
    bias: Optional[List[float]] = None
    zoom: Optional[List[float]] = field(default=(0.9, 1, 1.1))
    gaussian_noise: Optional[List[float]] = field(default=(0.01, 0.1))
    random_crop: Optional[List[int]] = None
    orig_img_shape: Optional[List[int]] = field(default=(256, 256, 3))


@dataclass
class OutputConfig(BaseConfig):
    output_dir: str = None
    save_model: bool = True
    save_mislabeled: bool = False


@dataclass
class MisoConfig(BaseConfig):
    name: str = ""
    description: str = ""
    cnn: ModelConfig = default_field(ModelConfig())
    dataset: DatasetConfig = default_field(DatasetConfig())
    training: TrainingConfig = default_field(TrainingConfig())
    augmentation: AugmentationConfig = default_field(AugmentationConfig())
    output: OutputConfig = default_field(OutputConfig())

    def sanitise(self):
        if self.name == "":
            self.name = self.dataset.source[:64] + "_" + self.cnn.type
            self.name = re.sub('[^A-Za-z0-9]+', '-', self.name)
        if self.cnn.img_shape is None:
            if self.cnn.type.endswith("_tl"):
                shape = KERAS_MODEL_PARAMETERS[self.cnn.type.split('_')[0]].default_input_shape
            else:
                if self.cnn.type.startswith("base_cyclic") or self.cnn.type.startswith("resnet_cyclic"):
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
        elif self.cnn.type.startswith("base_cyclic") or self.cnn.type.startswith("resnet_cyclic"):
            pass
        else:
            self.cnn.img_shape[2] = 3


def get_default_shape(cnn_type):
    if cnn_type.endswith("_tl"):
        return KERAS_MODEL_PARAMETERS[cnn_type.split('_')[0]].default_input_shape
    else:
        return [224, 224, None]


if __name__ == "__main__":
    m = MisoConfig()
    m.name = "test"
    m.description = "test description"
    print(m.dumps())
    xml = xmltodict.unparse({"config": m.asdict()}, pretty=True)
    print(xml)
    pprint(xmltodict.parse(xml))
    m = MisoConfig.load(xmltodict.parse(xml)["config"])
    pprint(m)
    # print(dicttoxml.dicttoxml(m.asdict(), attr_type=False, custom_root="config", item_func=lambda x: x[0]).toprettyxml())
    # xml = parseString(dicttoxml.dicttoxml(m.asdict())).toprettyxml()
    # print(xml)
    #
    # print(xmltodict.parse(xml))







