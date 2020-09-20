# Load TensorFlow
import tensorflow as tf
import os
import numpy as np
import skimage.io as skio
import skimage.transform as skt
import tensorflow as tf
from tensorflow.keras.utils import Sequence, OrderedEnqueuer
import numpy as np
from miso.data.datasource import DataSource
import lxml.etree as ET
from tensorflow.python.platform import gfile
import tensorflow.keras.backend as K
import os
import pandas as pd
import multiprocessing
from pathlib import Path


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

# ------------------------------------------------------------------------------------------------------------------- *
frozen_model = r"D:\Datasets\Seagrass\TrainedModels\ResNet18_20200701-190950\model\frozen_model.pb"
save_dir = r"D:\Datasets\Seagrass\TrainedModels\ResNet18_20200701-190950\model"
input_tensors = ['data']
output_tensors = ['softmax/Softmax']

# Set up the converter
converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_model,
                                                           input_tensors,
                                                           output_tensors)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Perform conversion and output file
tflite_quant_model = converter.convert()
open(os.path.join(save_dir, "converted_model.tflite"), "wb").write(tflite_quant_model)

# ------------------------------------------------------------------------------------------------------------------- *
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(os.path.join(save_dir, "converted_model.tflite"))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

session, input, output = load_from_frozen(frozen_model, input_tensors[0] + ":0", output_tensors[0] + ":0")
#
# im = skio.imread(r"D:\Datasets\Seagrass\SeagrassFramesPatches\Ferny - dense\7FEB20-DSC00051-042.jpg").astype(np.float32)
# im = skt.resize(im, input_details[0]['shape'][1:3]) / 255
# # im = im.astype(np.uint8)
# interpreter.set_tensor(input_details[0]['index'], im[np.newaxis, ...])
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.round(1))
# print(session.run(output, feed_dict={input: im[np.newaxis, ...]}).round(1))
#
# # Test with image
# im = skio.imread(r"D:\Datasets\Seagrass\SeagrassFramesPatches\Round - dense\8FEB20-DSC01455-013.jpg").astype(np.float32)
# im = skt.resize(im, input_details[0]['shape'][1:3]) / 255
# # im = im.astype(np.uint8)
# interpreter.set_tensor(input_details[0]['index'], im[np.newaxis, ...])
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.round(1))
# print(session.run(output, feed_dict={input: im[np.newaxis, ...]}).round(1))
#
# im = skio.imread(r"D:\Datasets\Seagrass\SeagrassFramesPatches\Strappy - dense\8FEB20-DSC00248-014.jpg").astype(np.float32)
# im = skt.resize(im, input_details[0]['shape'][1:3]) / 255
# # im = im.astype(np.uint8)
# interpreter.set_tensor(input_details[0]['index'], im[np.newaxis, ...])
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.round(1))
# print(session.run(output, feed_dict={input: im[np.newaxis, ...]}).round(1))
#
# im = skio.imread(r"D:\Datasets\Seagrass\SeagrassFramesPatches\Substrate\10FEB20-DSC03788-027.jpg").astype(np.float32)
# im = skt.resize(im, input_details[0]['shape'][1:3]) / 255
# # im = im.astype(np.uint8)
# interpreter.set_tensor(input_details[0]['index'], im[np.newaxis, ...])
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.round(1))
# print(session.run(output, feed_dict={input: im[np.newaxis, ...]}).round(1))


ds = DataSource()
ds.set_source(r"D:\Datasets\Seagrass\project_seagrass_multiscale_128_24000.xml", 40)
ds.use_mmap = True
ds.mmap_directory = r'D:\Temp'
ds.load_dataset((128,128), img_type='rgb', dtype=np.float32)
ds.split(0.2)

cls = []
for i, im in enumerate(ds.test_images):
    print("\r{}".format(np.round(i*100/len(ds.test_images)), 2), end='')
    interpreter.set_tensor(input_details[0]['index'], im[np.newaxis, ...])
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    cls.append(np.argmax(pred))
acc_all = np.sum(cls == ds.test_cls) / len(ds.test_cls)


tcls = []
for i, im in enumerate(ds.test_images):
    print("\r{}".format(np.round(i*100/len(ds.test_images)), 2), end='')
    pred = session.run(output, feed_dict={input: im[np.newaxis, ...]})
    tcls.append(np.argmax(pred))
acc_test = np.sum(tcls == ds.test_cls) / len(ds.test_cls)

print("Accuracy TF lite. All: {}, Test: {}".format(acc_all, acc_test))