import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Activation, \
    Lambda, UpSampling2D, \
                                    Conv2DTranspose, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy

from miso.layers import cyclic
from miso.archive.datasource import DataSource
import numpy as np


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def sampling2(args):
    z_mean, z_log_sigma, n_size = args
    epsilon = K.random_normal(shape=(n_size,), mean=0, stddev=1)
    return z_mean + K.exp(z_log_sigma/2) * epsilon


def vae_model(input_shape):
    # network parameters
    # input_shape = (image_size, image_size, 1)
    # batch_size = 128
    blocks = 2
    kernel_size = 5
    filters = 16
    latent_dim = 2
    # epochs = 30

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(blocks):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(blocks):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    return vae, z_mean, z_log_var





def build_conv_vae(input_shape, bottleneck_size, samp):
    # ENCODER
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Latent Variable Calculation
    shape = K.int_shape(x)
    flatten_1 = Flatten()(x)
    dense_1 = Dense(bottleneck_size, name='z_mean')(flatten_1)
    z_mean = BatchNormalization()(dense_1)
    print(dense_1)
    print(z_mean)
    flatten_2 = Flatten()(x)
    dense_2 = Dense(bottleneck_size, name='z_log_sigma')(flatten_2)
    z_log_sigma = BatchNormalization()(dense_2)
    print(dense_2)
    print(z_log_sigma)
    z = Lambda(samp)([z_mean, z_log_sigma])
    encoder = Model(input, [z_mean, z_log_sigma, z], name='encoder')

    # DECODER
    latent_input = Input(shape=(bottleneck_size,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_input)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    # x = Cropping2D([[0, 0], [0, 1]])(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # x = Cropping2D([[0, 1], [0, 1]])(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # x = Cropping2D([[0, 1], [0, 1]])(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    output = Conv2DTranspose(1, (3, 3), activation='tanh', padding='same')(x)
    decoder = Model(latent_input, output, name='decoder')

    print(encoder.outputs)
    print(input)
    print(decoder(encoder.outputs[2]))
    output_2 = decoder(encoder.outputs[2])
    vae = Model(input, output_2, name='vae')
    return vae, encoder, decoder, z_mean, z_log_sigma


# def vae_loss(input_img, output, vae=None):
    # Compute error in reconstruction
    # reconstruction_loss = mse(K.flatten(input_img), K.flatten(output))
    #
    # # Compute the KL Divergence regularization term
    # kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    #
    # # Return the average loss over all images in batch
    # total_loss = (reconstruction_loss + 0.0001 * kl_loss)





    # return total_loss


if __name__ == '__main__':

    input_shape = (64, 64, 1)
    bottleneck_size = 128
    image_size = 64
    # vae, z_mean, z_log_var = vae_model((64,64,1))

    vae, encoder, decoder, z_mean, z_log_sigma = \
        build_conv_vae(input_shape, bottleneck_size, sampling)

    # VAE loss = mse_loss or xent_loss + kl_loss
    # if False:
    #     reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    # else:
    # reconstruction_loss = binary_crossentropy(K.flatten(vae.inputs),
    #                                           K.flatten(vae.outputs))
    #
    # reconstruction_loss *= image_size * image_size
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    # vae.add_loss(vae_loss)
    # vae.compile(optimizer='rmsprop')
    # vae.summary()

    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

    reconstruction_loss = binary_crossentropy(K.flatten(vae.inputs),
                                              K.flatten(vae.outputs))
    reconstruction_loss = mse(K.flatten(vae.inputs),
                              K.flatten(vae.outputs))
    reconstruction_loss *= image_size * image_size
    # kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + 0.1 * kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    encoder.add_loss(vae_loss)
    encoder.compile(optimizer='rmsprop')
    # decoder.compile(optimizer='rmsprop', loss=vae_loss)

    input_source = r'C:\Users\rossm\Documents\Data\Foraminifera\ForamA\project_2.xml'

    data_source = DataSource()
    data_source.use_mmap = False
    data_source.set_source(input_source, 40)
    data_source.load_dataset(img_size=(image_size, image_size),
                             prepro_type=None,
                             prepro_params=(255, 0, 1),
                             img_type='greyscale',
                             print_status=True,
                             dtype=np.float32)
    data_source.split(0.25, 0, 0)

    # if args.weights:
    #     vae.load_weights(args.weights)
    # else:
    #     # train the autoencoder
    vae.fit((data_source.train_images, data_source.train_images),
            epochs=200,
            batch_size=32,
            validation_data=(data_source.test_images, None))
        # vae.save_weights('vae_cnn_mnist.h5')

    # plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")


















def base_cyclic_auto_encoder(input_shape,
                             filters=4,
                             blocks=4,
                             dropout=0.5,
                             dense=512,
                             conv_padding='same',
                             conv_activation='relu',
                             use_batch_norm=True,
                             global_pooling=None):

    inputs = Input(shape=input_shape)
    x = cyclic.CyclicSlice4()(inputs)
    for i in range(blocks):
        conv_filters = filters * 2 ** i
        # First layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        x = Activation(conv_activation)(x)
        # Second layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        x = Activation(conv_activation)(x)
        # Pool
        x = MaxPooling2D()(x)
        # Roll
        x = cyclic.CyclicRoll4()(x)
    # if global_pooling == 'avg':
    #     x = GlobalAveragePooling2D()(x)
    # elif global_pooling == 'max':
    #     x = GlobalMaxPooling2D()(x)
    # Dense layers
    encoded = x

    for i in range(blocks):
        conv_filters = filters * 2 ** i
        # First layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        x = Activation(conv_activation)(x)
        # Second layer
        x = Conv2D(conv_filters, (3, 3), padding=conv_padding, activation=None, kernel_initializer='he_normal')(x)
        x = Activation(conv_activation)(x)
        # Pool
        x = MaxPooling2D()(x)
        # Roll
        x = cyclic.CyclicRoll4()(x)

    # x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    # x = Dropout(dropout)(x)
    # x = Dense(dense, activation='relu')(x)
    # x = cyclic.CyclicDensePool4(pool_op=tf.reduce_mean)(x)
    # x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs, x, name='base_cyclic_auto_encoder')
    return model
