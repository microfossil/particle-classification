"""
Train an image classifier on the deep weeds dataset
"""

from miso.training.parameters import MisoConfig
from miso.training.trainer import train_image_classification_model

tp = MisoConfig()

tp.name = "ZooLake"
tp.description = "ZooLake dataset trained using MISO code"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Source directory (local folder or download link to dataset)
tp.dataset.source = r"C:\Users\ross.marchant\Downloads\data (1)\data\project.xml"
# Minimum number of images to include in a class
tp.dataset.min_count = 10
# Whether to map the images in class with not enough examples to an "others" class
tp.dataset.map_others = False
# Fraction of dataset used for validation
tp.dataset.train_split = 0.4
tp.dataset.val_split = 0.1
# Random seed used to split the dataset into train and validation
tp.dataset.random_seed = 0
# Set to a local directory to stored the loaded dataset on disk instead of in memory
tp.dataset.memmap_directory = None

# -----------------------------------------------------------------------------
# CNN
# -----------------------------------------------------------------------------
# CNN type
# Transfer learning:
# - resnet50_tl
# - resnet50_cyclic_tl
# - resnet50_cyclic_gain_tl
# Full network (custom)
# - base_cyclic
# - resnet_cyclic
# Full network (keras applications / qubvel image_classifiers)
# - resnet[18,34,50]
# - vgg[16,19]
# - efficientnetb[0-7]
tp.cnn.type = "efficientnetb2"
# Input image shape, set to None to use default size ([128, 128, 1] for custom, [224, 224, 3] for others)
tp.cnn.img_shape = [128, 128, 3]
# Input image colour space [greyscale/rgb]
tp.cnn.img_type = "rgb"
# Number of filters in first block (custom networks)
tp.cnn.filters = 4
# Number of blocks (custom networks), set to None for automatic selection
tp.cnn.blocks = None
# Size of dense layers (custom networks / transfer learning) as a list, e.g. [512, 512] for two dense layers size 512
tp.cnn.dense = None
# Whether to use batch normalisation
tp.cnn.use_batch_norm = True
# Type of pooling [avg, max, none]
tp.cnn.global_pooling = None
# Type of activation
tp.cnn.activation = "relu"
# Use A-Softmax
tp.cnn.use_asoftmax = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
# Number of images for each training step
tp.training.batch_size = 64
# Number of epochs after which training is stopped regardless
tp.training.max_epochs = 10000
# Number of epochs to monitor for no improvement by the adaptive learning rate scheduler.
# After no improvement for this many epochs, the learning rate is dropped by half
tp.training.alr_epochs = 40
# Number of learning rate drops after which training is suspended
tp.training.alr_drops = 4
# Monitor the validation loss instead?
tp.training.monitor_val_loss = False
# Use class weighting?
tp.training.use_class_weights = True
# Use class balancing via random over sampling? (Overrides class weights)
tp.training.use_class_undersampling = False
# Use train time augmentation?
tp.training.use_augmentation = True

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------
# Setting depends on the size of list passed:
# - length 2, e.g. [low, high] = random value between low and high
# - length 3 or more, e.g. [a, b, c] = choose a random value from this list
# Rotation
tp.augmentation.rotation = [0, 360]
# Gain: I' = I * gain
tp.augmentation.gain = [0.8, 1, 1.2]
# Gamma: I' = I ^ gamma
tp.augmentation.gamma = [0.5, 1, 2]
# Bias: I' = I + bias
tp.augmentation.bias = None
# Zoom: I'[x,y] = I[x/zoom, y/zoom]
tp.augmentation.zoom = [0.9, 1, 1.1]
# Gaussian noise std deviation
tp.augmentation.gaussian_noise = [0.01, 0.1]
# The parameters for the following are not random
# Random crop, e.g. [224, 224, 3]
# If random crop is used, you MUST set the original image size that the crop is taken from
tp.augmentation.random_crop = None
tp.augmentation.orig_img_shape = [256, 256, 3]

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
# Directory to save output
tp.output.save_dir = "."
# Save model?
save_model = True
# Save the mislabelled image analysis?
tp.output.save_mislabeled = False




# Train the model!!!
if __name__ == "__main__":
    train_image_classification_model(tp)

