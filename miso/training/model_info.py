import datetime
from typing import List, Dict
from lxml import etree as ET
import tensorflow as tf
import os
from collections import OrderedDict


class ModelInfo:

    def __init__(self,
                 name: str,
                 description: str,
                 type: str,
                 date: datetime.datetime,
                 protobuf: str,
                 params: Dict,
                 inputs: OrderedDict,
                 outputs: OrderedDict,
                 labels: List[str],
                 prepro_name: str,
                 prepro_params: List,
                 source_data: str,
                 accuracy: float,
                 precision: float,
                 recall: float,
                 f1score: float):

        self.name = name
        self.description = description
        self.type = type
        self.date = date
        self.params = params
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.prepro_name = prepro_name
        self.prepro_params = prepro_params
        self.protobuf = protobuf
        self.source_data = source_data
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1score = f1score

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, 'wb')
        f.write(self.to_xml())
        f.close()

    def to_xml(self):
        root = ET.Element("network")
        ET.SubElement(root, "name").text = self.name
        ET.SubElement(root, "description").text = self.description
        ET.SubElement(root, "type").text = self.type
        ET.SubElement(root, "date").text = "{0:%Y-%m-%d_%H%M%S}".format(self.date)
        ET.SubElement(root, "protobuf").text = self.protobuf

        parent_node = ET.SubElement(root, "params")
        for key, value in self.params.items():
            ET.SubElement(parent_node, key).text = str(value)

        parent_node = ET.SubElement(root, "inputs")
        for name, tensor in self.inputs.items():
            node = ET.SubElement(parent_node, "input")
            ET.SubElement(node, "name").text = name
            ET.SubElement(node, "operation").text = tensor.op.name
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
            ET.SubElement(node, "operation").text = tensor.op.name
            ET.SubElement(node, "height").text = str(tensor.shape[1])
            if len(tensor.shape) > 2:
                ET.SubElement(node, "width").text = str(tensor.shape[2])
            else:
                ET.SubElement(node, "width").text = "0"
            if len(tensor.shape) > 3:
                ET.SubElement(node, "channels").text = str(tensor.shape[3])
            else:
                ET.SubElement(node, "channels").text = "0"

        parent_node = ET.SubElement(root, "labels")
        for idx, value in enumerate(self.labels):
            ET.SubElement(parent_node, "label").text = str(value)

        parent_node = ET.SubElement(root, "prepro")
        ET.SubElement(parent_node, "name").text = self.prepro_name
        parent_node = ET.SubElement(parent_node, "params")
        for idx, value in enumerate(self.prepro_params):
            ET.SubElement(parent_node, "param").text = str(value)

        ET.SubElement(root, "source_data").text = str(self.source_data)
        ET.SubElement(root, "accuracy").text = str(self.accuracy)
        ET.SubElement(root, "precision").text = str(self.precision)
        ET.SubElement(root, "recall").text = str(self.recall)
        ET.SubElement(root, "f1score").text = str(self.f1score)

        return ET.tostring(root, pretty_print=True)


if __name__ == "__main__":
    input = tf.zeros((1, 256, 128, 3), dtype=tf.float32)
    output = tf.zeros((1, 256), dtype=tf.int16)
    info = ModelInfo("test",
                     "test of the model info xml parsing",
                     "not a network",
                     datetime.datetime.now(),
                     "frozen_model.pb",
                     {"topo": "hello", "style": 12, "alist": [1, 2, 3, 4]},
                     [input],
                     [output],
                     ["label_1", "label_2"],
                     [255, 0, 1],
                     "",
                     99.0,
                     85.3,
                     45.2,
                     11.0)
    info.save("test_model_info.xml")
