from collections import OrderedDict
from typing import List
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
# from miso.save import freeze_graph
from miso.training.model_info import ModelInfo
import tensorflow.keras.backend as K
try:
    import keras.backend as J
except:
    pass


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def convert_to_inference_mode(model, model_factory, tf_keras=True):
    # Trick to get around bugs in tensorflow 1.14.0
    # https://github.com/tensorflow/tensorflow/issues/31331#issuecomment-518655879
    model.save_weights("weights.h5")
    if tf_keras is True:
        K.clear_session()
        K.set_learning_phase(0)
    else:
        J.clear_session()
        J.set_learning_phase(0)
    model = model_factory()
    model.load_weights("weights.h5")
    remove("weights.h5")
    return model


def freeze_or_save(model, save_dir, metadata: ModelInfo=None, freeze=True, tf_keras=True):

    if tf_keras:
        if metadata is not None:
            metadata_tensor = K.constant(metadata.to_xml(), name="metadata", dtype='string')
            model = Model(model.inputs[0], [model.outputs[0], metadata_tensor])
        # K.set_learning_phase(0)
        tf.saved_model.simple_save(K.get_session(),
                                   save_dir,
                                   inputs={"input": model.inputs[0]},
                                   outputs={"output": model.outputs[0]})
    else:
        if metadata is not None:
            metadata_tensor = J.constant(metadata.to_xml(), name="metadata", dtype='string')
            model = Model(model.inputs[0], [model.outputs[0], metadata_tensor])
        # J.set_learning_phase(0)
        tf.saved_model.simple_save(J.get_session(),
                                   save_dir,
                                   inputs={"input": model.inputs[0]},
                                   outputs={"output": model.outputs[0]})
    if freeze:
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


def load(source: str,
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
    names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for n in names:
        print(n)

    # Input / output tensors
    input = session.graph.get_tensor_by_name(input_tensor)
    output = session.graph.get_tensor_by_name(output_tensor)
    return session, input, output
