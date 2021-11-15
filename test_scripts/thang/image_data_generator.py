from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    dtype="float32",
    rescale=1. / 255,
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    channel_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

gen = train_gen.flow_from_directory(r"C:\Users\rossm\Documents\Data\Plankton\Cristele\images_20190724_153703")

im = next(iter(gen))