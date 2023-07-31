from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from lxml import etree as ET
import os
import numpy as np

from miso.training.parameters import MisoConfig
from miso.utils.base_config import BaseConfig


@dataclass(kw_only=True)
class Input(BaseConfig):
    name: str
    operation: str
    height: int
    width: int
    channels: int

@dataclass(kw_only=True)
class Output(BaseConfig):
    name: str
    operation: str
    height: int
    width: int
    channels: int


@dataclass(kw_only=True)
class Label(BaseConfig):
    name: str
    operation: str
    height: int
    width: int
    channels: int


@dataclass
class Load(BaseConfig):
    training_epochs: int
    training_time: float
    training_split: float
    training_time_per_image: float
    inference_time_per_image: float


@dataclass(kw_only=True)
class ModelInfo(BaseConfig):
    name: str
    description: str
    type: str
    date: datetime
    protobuf: str
    source_data: str
    source_size: str
    accuracy: float
    precision: float
    recall: float
    f1score: float
    support: float
    training_epochs: int
    training_time: float
    training_split: float
    inference_time_per_image: float
    params: MisoConfig
    inputs: List[Input]
    outputs: List[Output]
    data_source_name: str
    labels: List[str]
    counts: List[int]
    prepro_name: str
    prepro_params: List[float]
    version: str = "3.0"

    def save_to_xml(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, 'wb')
        f.write(self.to_xml())
        f.close()

    def _get_op_name(self, tensor):
        try:
            return tensor.op.name
        except:
            return tensor.name

    def to_xml(self):
        root = ET.Element("network", version=self.version)
        ET.SubElement(root, "name").text = self.name
        ET.SubElement(root, "description").text = self.description
        ET.SubElement(root, "type").text = self.type
        ET.SubElement(root, "date").text = "{0:%Y-%m-%d_%H%M%S}".format(self.date)
        ET.SubElement(root, "protobuf").text = self.protobuf

        parent_node = ET.SubElement(root, "params")
        for key, value in self.params.asdict().items():
            ET.SubElement(parent_node, key).text = str(value)

        parent_node = ET.SubElement(root, "inputs")
        for name, tensor in self.inputs.items():
            node = ET.SubElement(parent_node, "input")
            ET.SubElement(node, "name").text = name
            ET.SubElement(node, "operation").text = self._get_op_name(tensor)
            ET.SubElement(node, "height").text = str(tensor.shape[1])
            if len(tensor.shape) > 2:
                ET.SubElement(node, "width").text = str(tensor.shape[2])
            else:
                ET.SubElement(node, "width").text = "0"
            if len(tensor.shape) > 3:
                ET.SubElement(node, "channels").text = str(tensor.shape[3])
            else:
                ET.SubElement(node, "channels").text = "0"

        parent_node = ET.SubElement(root, "outputs")
        for name, tensor in self.outputs.items():
            node = ET.SubElement(parent_node, "output")
            ET.SubElement(node, "name").text = name
            ET.SubElement(node, "operation").text = self._get_op_name(tensor)
            ET.SubElement(node, "height").text = str(tensor.shape[1])
            if len(tensor.shape) > 2:
                ET.SubElement(node, "width").text = str(tensor.shape[2])
            else:
                ET.SubElement(node, "width").text = "0"
            if len(tensor.shape) > 3:
                ET.SubElement(node, "channels").text = str(tensor.shape[3])
            else:
                ET.SubElement(node, "channels").text = "0"

        ET.SubElement(root, "source_data").text = str(self.data_source_name)
        ET.SubElement(root, "source_size").text = str(np.sum(self.counts))
        parent_node = ET.SubElement(root, "labels")
        for idx, value in enumerate(self.labels):
            node = ET.SubElement(parent_node, "label")
            ET.SubElement(node, "code").text = value
            ET.SubElement(node, "count").text = str(self.counts[idx])
            ET.SubElement(node, "precision").text = str(self.precision[idx])
            ET.SubElement(node, "recall").text = str(self.recall[idx])
            ET.SubElement(node, "f1score").text = str(self.f1score[idx])
            ET.SubElement(node, "support").text = str(self.support[idx])

        parent_node = ET.SubElement(root, "prepro")
        ET.SubElement(parent_node, "name").text = self.prepro_name
        parent_node = ET.SubElement(parent_node, "params")
        for idx, value in enumerate(self.prepro_params):
            ET.SubElement(parent_node, "param").text = str(value)

        ET.SubElement(root, "accuracy").text = str(self.accuracy)
        ET.SubElement(root, "precision").text = str(np.mean(self.precision))
        ET.SubElement(root, "recall").text = str(np.mean(self.recall))
        ET.SubElement(root, "f1score").text = str(np.mean(self.f1score))

        parent_node = ET.SubElement(root, "load")
        ET.SubElement(parent_node, "training_epochs").text = str(self.training_epochs)
        ET.SubElement(parent_node, "training_time").text = str(self.training_time)
        ET.SubElement(parent_node, "training_split").text = str(self.training_split)
        ET.SubElement(parent_node, "training_time_per_image").text = str(self.training_time / self.training_epochs / (np.sum(self.counts) * (1 - self.training_split)))
        ET.SubElement(parent_node, "inference_time_per_image").text = str(np.mean(self.inference_time_per_image))

        return ET.tostring(root, pretty_print=True)
