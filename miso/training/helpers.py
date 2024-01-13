import tensorflow as tf
import numpy as np


def predict_in_batches(model, generator):
    results = []
    try:
        for x, y in iter(generator):
            results.append(model.predict(x))
    except tf.errors.OutOfRangeError:
        pass
    results = np.concatenate(results, axis=0)
    return results