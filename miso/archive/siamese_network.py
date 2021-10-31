from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
"""
Modified from https://keras.io/examples/mnist_siamese/
"""


def vector_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        print(d)
        for i in range(n):
            print(i)
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


if __name__ == "__main__":
    from miso.archive.datasource import DataSource
    from miso.models.transfer_learning import head

    # num_classes = 10
    epochs = 200

    # The data, split between train and test sets
    ds = DataSource()
    ds.set_source(r'D:\Datasets\Foraminifera\images_20200226_114546 Fifth with background', 40)
    ds.use_mmap = True
    ds.mmap_directory = r'D:\Temp'
    ds.load_dataset((224,224), img_type='greyscale3', dtype=np.float32)
    ds.split(0.2)

    tl = head('resnet50', (224, 224, 3))
    temp = []
    for i in np.arange(0, len(ds.train_images), 1000):
        print(i)
        temp.append(tl.predict(ds.train_images[i:i+1000]))
    ds.train_vectors = np.concatenate(temp, axis=0)
    temp = []
    for i in np.arange(0, len(ds.test_images), 1000):
        print(i)
        temp.append(tl.predict(ds.test_images[i:i + 1000]))
    ds.test_vectors = np.concatenate(temp, axis=0)

    # (x_train, y_train) = (np.reshape(ds.train_images, (len(ds.train_images), -1)), ds.train_cls)
    # (x_test, y_test) = (np.reshape(ds.test_images, (len(ds.test_images), -1)), ds.test_cls)
    (x_train, y_train) = (ds.train_vectors, ds.train_cls)
    (x_test, y_test) = (ds.test_vectors, ds.test_cls)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    input_shape = x_train.shape[1:]
    num_classes = ds.num_classes

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    # network definition
    base_network = vector_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(base_network.predict(ds.train_vectors), ds.train_cls)
    yc_pred = neigh.predict(base_network.predict(ds.test_vectors))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(ds.test_cls, yc_pred))

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(ds.train_vectors, ds.train_cls)
    yc_pred = neigh.predict(ds.test_vectors)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(ds.test_cls, yc_pred))