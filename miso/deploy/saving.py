import os
import shutil
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from miso.deploy.model_info import ModelInfo
import tensorflow.keras.backend as K
import tempfile
import lxml.etree as ET
import numpy as np


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


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


def save(model, save_dir):
    tf.saved_model.simple_save(K.get_session(),
                               save_dir,
                               inputs={"input": model.inputs[0]},
                               outputs={"output": model.outputs[0]})


def freeze(model, save_dir, metadata: ModelInfo=None):
    # if metadata is not None:
    #     metadata_tensor = K.constant(metadata.to_xml(), name="metadata", dtype='string')
    #     model = Model(model.inputs[0], [model.outputs[0], metadata_tensor])
    # K.set_learning_phase(0)

    # Save using simple save
    save(model, save_dir)
    # Freeze graph
    if metadata is not None:
        ext = ".pb"
    else:
        ext = ".pb"
    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[0].op.name,
                              None,
                              None,
                              os.path.join(save_dir, "frozen_model" + ext),
                              False,
                              "",
                              input_saved_model_dir=save_dir)
    # Delete saved models
    remove(os.path.join(save_dir, "saved_model.pb"))
    remove(os.path.join(save_dir, "variables"))
    # Save info
    metadata.save(os.path.join(save_dir, "network_info.xml"))


def load_from_frozen(source: str,
                     input_tensor="input_1:0",
                     output_tensor="conv2d_23/Sigmoid:0",
                     session=None):

    if not os.path.exists(source):
        raise FileNotFoundError('The graph file was not found on the system.\nThe path was: ' + source)

    # Load network
    if session is None:
        session = K.get_session()
    with gfile.Open(source, 'rb') as f:
        graph_def = tf.GraphDef()
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
    print("protobuf: " + protobuf)

    input = None
    output = None
    img_size = np.zeros(3, dtype=np.int)
    cls_labels = []

    list_xml = project.find('labels')
    for i, entry_xml  in enumerate(list_xml.iter('label')):
        code = entry_xml.find('code').text
        cls_labels.append(code)

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

    full_protobuf_path = os.path.join(os.path.dirname(filename), protobuf)
    session, input, output = load_from_frozen(full_protobuf_path, input_name, output_name)
    return session, input, output, img_size, cls_labels

    # if not os.path.exists(filename):
    #     raise FileNotFoundError('The xml information file was not found on the system.\nThe path was: ' + filename)
    #
    # # Load network
    # if session is None:
    #     session = K.get_session()
    # with gfile.Open(source, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     session.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')
    #
    # # Information
    # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for n in names:
    #     print(n)
    #
    # # Input / output tensors
    # input = session.graph.get_tensor_by_name(input_tensor)
    # output = session.graph.get_tensor_by_name(output_tensor)
    # return session, input, output
