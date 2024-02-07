import datetime
from typing import List, Dict
from lxml import etree as ET
import os
import numpy as np
from collections import OrderedDict

from miso.training.parameters import MisoParameters


class ModelInfo:
    def __init__(self,
                 name: str,
                 description: str,
                 type: str,
                 date: datetime.datetime,
                 protobuf: str,
                 params: MisoParameters,
                 inputs: OrderedDict,
                 outputs: OrderedDict,
                 data_source_name: str,
                 labels: List[str],
                 counts: List[int],
                 prepro_name: str,
                 prepro_params: List,
                 accuracy: float,
                 precision: float,
                 recall: float,
                 f1score: float,
                 support: float,
                 training_epochs: int,
                 training_time: float,
                 training_split: float,
                 inference_time_per_image: float):

        self.name = name
        self.description = description
        self.type = type
        self.date = date
        self.params = params
        self.inputs = inputs
        self.outputs = outputs
        self.data_source_name = data_source_name
        self.labels = labels
        self.counts = counts
        self.prepro_name = prepro_name
        self.prepro_params = prepro_params
        self.protobuf = protobuf
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1score = f1score
        self.support = support
        self.training_epochs = training_epochs
        self.training_time = training_time
        self.training_split = training_split
        self.inference_time_per_image = inference_time_per_image
        self.version = "2.1"

    def save(self, filename):
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

        # parent_node = ET.SubElement(root, "params")
        # for key, value in self.params.asdict().items():
        #     ET.SubElement(parent_node, key).text = str(value)

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
        ET.SubElement(parent_node, "training_time_per_image").text = str(
            self.training_time / self.training_epochs / (np.sum(self.counts) * (1 - self.training_split)))
        ET.SubElement(parent_node, "inference_time_per_image").text = str(np.mean(self.inference_time_per_image))

        return ET.tostring(root, pretty_print=True)

    @staticmethod
    def from_xml(xml_data_or_path):
        if os.path.exists(xml_data_or_path):
            tree = ET.parse(xml_data_or_path)
            root = tree.getroot()
        else:
            root = ET.fromstring(xml_data_or_path)

        name = root.findtext('name')
        description = root.findtext('description')
        model_type = root.findtext('type')
        date_str = root.findtext('date')
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d_%H%M%S")
        protobuf = root.findtext('protobuf')

        inputs = []
        for inp in root.find('inputs').findall('input'):
            d = {}
            d["name"] = inp.findtext('name')
            d["operation"] = inp.findtext('operation')
            d["height"] = inp.findtext('height')
            d["width"] = inp.findtext('width')
            d["channels"] = inp.findtext('channels')
            inputs.append(d)

        outputs = []
        for out in root.find('outputs').findall('output'):
            d = {}
            d["name"] = out.findtext('name')
            d["operation"] = out.findtext('operation')
            d["height"] = out.findtext('height')
            d["width"] = out.findtext('width')
            d["channels"] = out.findtext('channels')
            outputs.append(d)

        labels = [label.findtext('code') for label in root.findall('.//labels/label')]

        return ModelInfo(name,
                         description,
                         model_type,
                         date,
                         protobuf,
                         None,
                         inputs,
                         outputs,
                         None,
                         labels,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None)
