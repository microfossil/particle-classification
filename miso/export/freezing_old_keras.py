from typing import List
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
import keras.backend as K
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework import graph_io


def freeze_and_save_graph(model, save_dir):
    """
    Saves the network to a protobuf file on disk
    :param session: Tensorflow session
    :param save_dir: Directory to save to
    :param name: Name of the file to save to
    :return: None
    """
    # checkpoint_prefix = os.path.join(save_dir, "model")
    # checkpoint_state_name = "checkpoint"
    # input_graph_name = "model.pb"
    # output_graph_name = name
    #
    # session = K.get_session()
    K.set_learning_phase(0)
    #
    # saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    # checkpoint_path = saver.save(session, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
    # graph_io.write_graph(session.graph, save_dir, input_graph_name)
    #

    tf.saved_model.simple_save(K.get_session(),
                               save_dir,
                               inputs={"image": model.inputs[0]},
                               outputs={"mask": model.outputs[0]})

    # saver = tf.train.Saver(tf.trainable_variables())

    # checkpoint_path = saver.save(
    #     session,
    #     checkpoint_prefix,
    #     global_step=0,
    #     latest_filename=checkpoint_state_name)

    # tf.train.write_graph(session.graph, save_dir, input_graph_name, as_text=True)

    # input_graph_path = os.path.join(save_dir, input_graph_name)
    # input_saver_def_path = ""
    # input_binary = False
    # output_node_names = model.outputs[0].op.name
    # restore_op_name = "save/restore_all"
    # filename_tensor_name = "save/Const:0"
    # output_graph_path = os.path.join(save_dir, output_graph_name)
    # clear_devices = False
    # freeze_graph.freeze_graph(input_graph_path,
    #                           input_saver_def_path,
    #                           input_binary,
    #                           checkpoint_path,
    #                           output_node_names,
    #                           restore_op_name,
    #                           filename_tensor_name,
    #                           output_graph_path,
    #                           clear_devices,
    #                           "",
    #                           input_saved_model_dir=save_dir)
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


def load(source: str,
         input_tensor="input_1:0",
         mask_tensor="conv2d_23/Sigmoid:0"):
    """
    Load a network from a tensorflow graph ProtoBuf file
    :param source: The ProtoBuf file created with network.network.freeze_and_save_graph
    :param input_tensor: Input image tensor name
    :param input_vector_tensor: Input vector tensor name
    :param prob_tensor: Output class probabilities tensor name
    :param vector_tensor: Output vector tensor name
    :return: (session, input tensor, output probs tensor, output vector tensor)
    """
    if not os.path.exists(source):
        raise FileNotFoundError('The graph file was not found on the system.\nThe path was: ' + source)

    # Load network
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
    mask = session.graph.get_tensor_by_name(mask_tensor)
    return session, input, mask


def save_network_definition_xml(self,
                                input: tf.Tensor,
                                output: tf.Tensor,
                                vector: tf.Tensor,
                                labels: List[str],
                                range: int,
                                filename: str,
                                output_dir: str,
                                name: str):
    """
    Creates an xml file that describes the network
    Used to import it into ForamTrieur
    :param input: Input image tensor name
    :param output: Output classification probability tensor name
    :param vector: Output vector tensor name
    :param labels: List of class labels
    :param range: Expected input range of the images, e.g. 1 means we expect input in the range [0,1]
    :param filename: Name of XML file to write to
    :param output_dir: Directory to save XML file
    :return: None
    """
    root = ET.Element("network")
    input_dims = input.get_shape().as_list()
    ET.SubElement(root, "name").text = name
    ET.SubElement(root, "width").text = str(input_dims[1])
    ET.SubElement(root, "height").text = str(input_dims[2])
    ET.SubElement(root, "channels").text = str(input_dims[3])
    ET.SubElement(root, "range").text = str(range)
    ET.SubElement(root, "input").text = input.name
    ET.SubElement(root, "output").text = output.name
    ET.SubElement(root, "vector").text = vector.name
    ET.SubElement(root, "filename").text = filename
    labels_node = ET.SubElement(root, "labels")
    for label in labels:
        label_node = ET.SubElement(labels_node, "label")
        ET.SubElement(label_node, "code").text = str(label)
        ET.SubElement(label_node, "name")
        ET.SubElement(label_node, "description")
        ET.SubElement(label_node, "isMorphotype").text = "true"

    f = open(os.path.join(output_dir, name + ".xml"), 'wb')
    f.write(ET.tostring(root, pretty_print=True))
    f.close()
