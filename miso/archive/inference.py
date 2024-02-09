"""
Performs inference on a directory of images organised by subdirectories, returning a pandas dataframe with the filename
and its classification
"""
import tensorflow as tf
from keras.utils import Sequence, OrderedEnqueuer
import numpy as np
import lxml.etree as ET
from tensorflow.python.platform import gfile
import keras.backend as K
import os
import pandas as pd
import multiprocessing
from pathlib import Path


class InferenceGenerator(Sequence):
    def __init__(self, source_dir, batch_size, img_size, img_type):
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_type = img_type
        filenames = DataSource.parse_directory(source_dir)
        self.filenames = [v for key, val in filenames.items() for v in val]

    def __data_generation(self, filenames):
        images = []
        for filename in filenames:
            image = DataSource.load_image(filename, self.img_size, self.img_type)
            image = DataSource.preprocess_image(image)
            images.append(image)
        return filenames, np.asarray(images)

    def __getitem__(self, index):
        batch_filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        filenames_and_images = self.__data_generation(batch_filenames)
        return filenames_and_images

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))


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
    input = session.graph.get_tensor_by_name(input_tensor)
    output = session.graph.get_tensor_by_name(output_tensor)
    return session, input, output


def load_from_xml(filename, session=None):
    project = ET.parse(filename).getroot()

    protobuf = project.find('protobuf').text
    print("protobuf: " + protobuf)

    input = None
    output = None
    img_size = np.zeros(3, dtype=int)
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


def process(network_info, images_dir, output_dir, threshold=0.8):
    session, input_tensor, output_tensor, img_size, cls_labels = load_from_xml(network_info)
    print("Input tensor: {}".format(input_tensor))
    print("Output tensor: {}".format(output_tensor))
    print("Image size: {}".format(img_size))
    print("Classes: {}".format(cls_labels))
    print()
    print("Parsing source directory... (this can take some time)")
    if img_size[2] == 3:
        img_type = 'rgb'
    else:
        img_type = 'greyscale'
    gen = InferenceGenerator(images_dir, 64, img_size, img_type)

    print("Files: {}".format(len(gen.filenames)))
    print("Batches: {}".format(len(gen)))

    workers = np.min((multiprocessing.cpu_count(), 8))
    print("Workers: {}".format(workers))
    print()

    filenames = []
    cls_index = []
    cls_names = []
    score = []
    enq = OrderedEnqueuer(gen, use_multiprocessing=True)
    enq.start(workers=workers, max_queue_size=multiprocessing.cpu_count()*4)
    output_generator = enq.get()
    for i in range(len(gen)):
        print("\r{} / {}".format(i, len(gen)), end='')
        batch_filenames, batch_images = next(output_generator)
        result = session.run(output_tensor, feed_dict={input_tensor: batch_images})
        cls = np.argmax(result, axis=1)
        scr = np.max(result, axis=1)
        cls_name = [cls_labels[i] for i in cls]
        filenames.extend(batch_filenames)
        cls_index.extend(cls)
        cls_names.extend(cls_name)
        score.extend(scr)
    enq.stop()
    print()
    print("Done")
    print("See {} for results".format(output_dir))

    parents = [Path(f).parent.name for f in filenames]
    files = [Path(f).name for f in filenames]

    df = pd.DataFrame(data={'filename': filenames, 'parent': parents, 'file': files, 'class': cls_names, 'class_index': cls_index, 'score': score})
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "inference.csv"))


if __name__ == "__main__":
    network_info = r"D:\Development\Microfossil\PaperExperiments\Paper1CNNForMicrofossil\MD972138\Output\MD972138_base_cyclic_8_20200410-173126\model\network_info.xml"
    images_directory = r"D:\Datasets\Foraminifera\images_20200226_114546 Fifth with background"
    output_directory = r"D:\output"
    process(network_info, images_directory, output_directory)

