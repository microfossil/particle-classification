import ast
import os
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from miso.deploy.model_info import ModelInfo
import tensorflow.keras.backend as K
import tempfile
import lxml.etree as ET
import numpy as np
import tf2onnx


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    # else:
    #     raise ValueError("file {} is not a file or dir.".format(path))


def convert_to_inference_mode(model, model_factory):
    # Trick to get around bugs in tensorflow 1.14.0
    # https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
    with tempfile.TemporaryDirectory() as dirpath:
        weights_filename = os.path.join(dirpath, "weights.h5")
        model.save_weights(weights_filename)
        K.clear_session()
        K.set_learning_phase(0)
        model = model_factory()
        model.load_weights(weights_filename)
        remove(weights_filename)
    return model


def convert_to_inference_mode_tf2(model, model_factory):
    # Trick to get around bugs in tensorflow 1.14.0
    # https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
    with tempfile.TemporaryDirectory() as dirpath:
        weights_filename = os.path.join(dirpath, "weights.tf")
        model.save_weights(weights_filename)
        K.clear_session()
        K.set_learning_phase(0)
        model = model_factory()
        model.load_weights(weights_filename)
        remove(weights_filename)
    return model


def save(model, save_dir):
    tf.saved_model.simple_save(K.get_session(),
                               save_dir,
                               inputs={"input": model.inputs[0]},
                               outputs={"output": model.outputs[0]})


def freeze(model, save_dir):
    # if metadata is not None:
    #     metadata_tensor = K.constant(metadata.to_xml(), name="metadata", dtype='string')
    #     model = Model(model.inputs[0], [model.outputs[0], metadata_tensor])
    # K.set_learning_phase(0)

    # Save using simple save
    save(model, save_dir)
    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[0].op.name,
                              None,
                              None,
                              os.path.join(save_dir, "frozen_model.pb"),
                              False,
                              "",
                              input_saved_model_dir=save_dir)
    # Delete saved models
    remove(os.path.join(save_dir, "saved_model.pb"))
    remove(os.path.join(save_dir, "variables"))

def load_frozen_model(source: str,
                      input_tensor="input_1:0",
                      output_tensor="conv2d_23/Sigmoid:0",
                      session=None):
    if not os.path.exists(source):
        raise FileNotFoundError('The graph file was not found on the system.\nThe path was: ' + source)
    # Load network
    if session is None:
        session = K.get_session()
    with gfile.Open(source, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        session.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    # Information
    # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for n in names:
    #     print(n)
    # Input / output tensors
    input = session.graph.get_tensor_by_name(input_tensor)
    output = session.graph.get_tensor_by_name(output_tensor)
    return session, input, output


def load_from_xml(filename, session=None):
    project = ET.parse(filename).getroot()

    protobuf = project.find('protobuf').text
    print(f"Loading model from {filename}")
    print("- protobuf: " + protobuf)

    input = None
    output = None
    img_size = np.zeros(3, dtype=int)
    cls_labels = []

    list_xml = project.find('labels')
    print("- labels:")
    for i, entry_xml in enumerate(list_xml.iter('label')):
        code = entry_xml.find('code').text
        cls_labels.append(code)
        print(f"  - {code}")

    list_xml = project.find('inputs')
    for i, entry_xml in enumerate(list_xml.iter('input')):
        if i == 0:
            input_name = entry_xml.find('operation').text + ":0"
            img_size[0] = int(entry_xml.find('height').text)
            img_size[1] = int(entry_xml.find('width').text)
            img_size[2] = int(entry_xml.find('channels').text)

    list_xml = project.find('outputs')
    for i, entry_xml in enumerate(list_xml.iter('output')):
        if i == 0:
            output_name = entry_xml.find('operation').text + ":0"

    # Find the 'cnn' tag within 'params'
    cnn_tag = project.find('.//params/cnn')
    img_type = "rgb"
    if cnn_tag is not None and cnn_tag.text:
        txt = cnn_tag.text
        if "'k'" in txt or "'greyscale'" in txt:
            img_type = "k"

    full_protobuf_path = os.path.join(os.path.dirname(filename), protobuf)

    print(f"- input: {input_name}")
    print(f"  - height: {img_size[0]}")
    print(f"  - width: {img_size[1]}")
    print(f"  - channels: {img_size[2]}")
    print(f"- output: {output_name}")

    model = load_frozen_model_tf2(full_protobuf_path, input_name, output_name)
    return model, img_size, img_type, cls_labels


import onnxruntime as rt

def load_onnx_from_xml(filename):
    project = ET.parse(filename).getroot()

    protobuf = str(list(Path(filename).parent.glob("*.onnx"))[0])
    print(f"Loading model from {filename}")
    print(f"- protobuf: {protobuf}")

    input = None
    output = None
    img_size = np.zeros(3, dtype=int)
    cls_labels = []

    list_xml = project.find('labels')
    print("- labels:")
    for i, entry_xml in enumerate(list_xml.iter('label')):
        code = entry_xml.find('code').text
        cls_labels.append(code)
        print(f"  - {code}")

    list_xml = project.find('inputs')
    for i, entry_xml in enumerate(list_xml.iter('input')):
        if i == 0:
            input_name = entry_xml.find('operation').text + ":0"
            img_size[0] = int(entry_xml.find('height').text)
            img_size[1] = int(entry_xml.find('width').text)
            img_size[2] = int(entry_xml.find('channels').text)

    list_xml = project.find('outputs')
    for i, entry_xml in enumerate(list_xml.iter('output')):
        if i == 0:
            output_name = entry_xml.find('operation').text + ":0"

    # Find the 'cnn' tag within 'params'
    cnn_tag = project.find('.//params/cnn')
    img_type = "rgb"
    if cnn_tag is not None and cnn_tag.text:
        txt = cnn_tag.text
        if "'k'" in txt or "'greyscale'" in txt:
            img_type = "k"

    full_protobuf_path = os.path.join(os.path.dirname(filename), protobuf)

    print(f"- input: {input_name}")
    print(f"  - height: {img_size[0]}")
    print(f"  - width: {img_size[1]}")
    print(f"  - channels: {img_size[2]}")
    print(f"- output: {output_name}")


    sess = rt.InferenceSession(str(protobuf))

    return sess, img_size, img_type, cls_labels




"""
Adapted from https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
"""
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def wrap_frozen_graph_tf2(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def save_frozen_model_tf2(model, out_dir, filename):
    print("-" * 80)
    print("Freezing model")
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # layers = [op.name for op in frozen_func.graph.get_operations()]
    # for layer in layers:
    #     print(layer)

    print("- model inputs: {}".format(frozen_func.inputs))
    print("- model outputs: {}".format(frozen_func.outputs))

    # Save frozen graph from frozen ConcreteFunction to hard drive
    # logdir="./frozen_models",
    print(out_dir)
    print(filename)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=os.path.join(out_dir),
                      name=filename,
                      as_text=False)
    return frozen_func


def save_model_as_onnx(model, input_name, input_shape, out_dir, opset=13):
    spec = (tf.TensorSpec(input_shape, tf.float32, name=input_name),)
    output_path = os.path.join(out_dir, "model.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)


def load_frozen_model_tf2(filepath, inputs, outputs):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph_tf2(graph_def=graph_def,
                                        inputs=inputs,
                                        outputs=outputs,
                                        print_graph=False)
    # print("-" * 80)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)
    return frozen_func
