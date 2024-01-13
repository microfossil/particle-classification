"""
Training parameters
"""
import copy

from pathlib import Path
from typing import Union, List, Optional
from dataclasses import dataclass, field
import re
from marshmallow_dataclass import class_schema

from miso.models.keras_models import KERAS_MODEL_PARAMETERS
from miso.utils.compact_json import CompactJSONEncoder


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))

@dataclass
class BaseParameters(object):
    class Meta:
        ordered = True

    def dumps(self):
        return class_schema(self.__class__)().dumps(self, indent=4, cls=CompactJSONEncoder)

    def save(self, path: str):
        with open(path, "w") as fp:
            metadata = class_schema(self.__class__)().dumps(self, indent=4)
            fp.write(metadata)

    @classmethod
    def loads(cls, json):
        instance = class_schema(cls)().loads(json)
        return instance
    @classmethod
    def load(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        instance = class_schema(cls)().loads(path.read_text())
        return instance


@dataclass
class ModelParameters(BaseParameters):
    # Common values
    id: str = "base_cyclic"
    img_shape: List[int] = field(default=(128, 128, 1))
    img_type: str = "greyscale"

    # Transfer learning cyclic layers
    use_cyclic: bool = False
    use_cyclic_gain: bool = False

    # Parameters for custom networks
    filters: Optional[int] = 4
    blocks: Optional[int] = None
    dense: Optional[int] = None
    use_batch_norm: bool = True
    global_pooling: bool = None
    activation: str = "relu"
    use_asoftmax: bool = False


@dataclass
class TrainingParameters(BaseParameters):
    batch_size: int = 64
    max_epochs: int = 10000
    alr_epochs: int = 10
    alr_drops: int = 4
    monitor_val_loss: bool = False
    use_class_weights: bool = True
    use_class_undersampling: bool = False
    use_augmentation: bool = True
    use_transfer_learning: bool = False
    transfer_learning_augmentation_factor: int = 0


@dataclass
class OptimizerParameters(BaseParameters):
    name: str = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9
    decay: float = 0.0
    nesterov: bool = False


@dataclass
class DatasetParameters(BaseParameters):
    num_classes: Optional[int] = None
    source: Optional[str] = None
    min_count: Optional[int] = 10
    train_split: Optional[float] = None
    val_split: Optional[float] = 0.2
    map_others: bool = False
    random_seed: int = 0
    memmap_directory: str = None


@dataclass
class AugmentationParameters(BaseParameters):
    rotation: List[float] = field(default=(0, 360))
    gain: Optional[List[float]] = field(default=(0.8, 1, 1.2))
    gamma: Optional[List[float]] = field(default=(0.5, 1, 2))
    bias: Optional[List[float]] = None
    zoom: Optional[List[float]] = field(default=(0.9, 1, 1.1))
    gaussian_noise: Optional[List[float]] = field(default=(0.01, 0.1))
    random_crop: Optional[List[int]] = None
    orig_img_shape: Optional[List[int]] = field(default=(256, 256, 3))


@dataclass
class OutputParameters(BaseParameters):
    save_dir: str = None
    save_model: bool = True
    save_mislabeled: bool = False


@dataclass
class MisoParameters(BaseParameters):
    name: str = ""
    description: str = ""
    cnn: ModelParameters = default_field(ModelParameters())
    dataset: DatasetParameters = default_field(DatasetParameters())
    training: TrainingParameters = default_field(TrainingParameters())
    augmentation: AugmentationParameters = default_field(AugmentationParameters())
    output: OutputParameters = default_field(OutputParameters())
    optimizer: OptimizerParameters = default_field(OptimizerParameters())

    def sanitise(self):
        if self.name == "":
            self.name = self.dataset.source[:64] + "_" + self.cnn.id
            self.name = re.sub('[^A-Za-z0-9]+', '-', self.name)
        if self.cnn.img_shape is None:
            if self.cnn.id.endswith("_tl"):
                shape = KERAS_MODEL_PARAMETERS[self.cnn.id.split('_')[0]].default_input_shape
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
        elif self.cnn.id.startswith("base_cyclic") or self.cnn.id.startswith("resnet_cyclic"):
            pass
        else:
            self.cnn.img_shape[2] = 3


def get_default_shape(cnn_type):
    if cnn_type.endswith("_tl"):
        return KERAS_MODEL_PARAMETERS[cnn_type.split('_')[0]].default_input_shape
    else:
        return [224, 224, None]


if __name__ == "__main__":
    m = MisoParameters()
    print(m.dumps())







