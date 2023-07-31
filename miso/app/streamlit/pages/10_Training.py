import os.path
from pathlib import Path

import streamlit as st
from miso.app.tasks.gpu_tasks import train_model_task

from miso.models.keras_models import KERAS_MODEL_PARAMETERS
from miso.training.parameters import MisoConfig

if "config" not in st.session_state:
    config = MisoConfig()
else:
    config = st.session_state.config

st.title("Training")


st.markdown(
    """
    ### Name
    """
)

config.name = st.text_input("Name (leave blank to automatically configure)", value=config.name)
config.description = st.text_area("Description", value=config.description)

st.markdown(
    """
    ### Dataset    
    """
)

config.dataset.source = st.text_input(
    "Path to PT project on server, e.g. /home/USERNAME/Documents/forams/project.xml"
)
if os.path.exists(config.dataset.source):
    st.success("Path is valid")
else:
    st.error("Path is invalid")

config.dataset.min_count = st.number_input("Minimum number of images per class",
                                        value=config.dataset.min_count,
                                        min_value=0)
config.dataset.val_split = st.number_input("Fraction of images to use for validation",
                                        value=config.dataset.val_split,
                                        min_value=0.0,
                                        max_value=1.0)
config.dataset.map_others = st.checkbox("Map all classes with fewer than the minimum number of images to 'other'",
                                        value=config.dataset.map_others)
config.dataset.random_seed = st.number_input("Random seed used for splitting, keep constant to have the same split each time", value=config.dataset.random_seed, min_value=0)

st.markdown(
    """
    ### Model    
    """
)




config.training.use_transfer_learning = st.checkbox("Use transfer learning (fast training)", value=config.training.use_transfer_learning)

typical_sizes = [32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512]

if config.training.use_transfer_learning:
    config.cnn.type = "resnet50"
    cnn_types = list(KERAS_MODEL_PARAMETERS.keys())
    config.cnn.type = st.selectbox("CNN type", cnn_types, index=cnn_types.index(config.cnn.type))
    config.training.transfer_learning_augmentation_factor = st.number_input("Number of augmentations per image to create (0 recommened for large datasets)", value=config.training.transfer_learning_augmentation_factor, min_value=0, step=1)
    config.cnn.use_tl_cyclic = st.checkbox("Use cyclic layers with transfer learning", value=config.cnn.use_tl_cyclic)
    if config.cnn.use_tl_cyclic:
        config.cnn.use_tl_cyclic_gain = st.checkbox("Use gain layers with transfer learning", value=config.cnn.use_tl_cyclic_gain)
    default_size = KERAS_MODEL_PARAMETERS[config.cnn.type][2][0]
    config.cnn.img_type = "rgb"
    # default_size = KERAS_MODEL_PARAMETERS[config.cnn.type].default_input_shape[0]
    # st.write("Input size: ", default_size)
else:
    config.cnn.type = "base_cyclic"
    cnn_types = ["base_cyclic", "resnet_cyclic"]
    cnn_types.extend(KERAS_MODEL_PARAMETERS.keys())
    config.cnn.type = st.selectbox("CNN type", cnn_types, index=cnn_types.index(config.cnn.type))

    if config.cnn.type == "base_cyclic" or config.cnn.type == "resnet_cyclic":
        filters = [4, 8, 16, 32]
        config.cnn.filters = st.selectbox("Number of filters", filters, index=filters.index(config.cnn.filters))
        default_size = 128
    else:
        default_size = KERAS_MODEL_PARAMETERS[config.cnn.type][2][0]

input_size = st.selectbox("Input image size", typical_sizes, index=typical_sizes.index(default_size))
config.cnn.img_type = st.selectbox("Image type", ["greyscale", "rgb"], index=["greyscale", "rgb"].index(config.cnn.img_type))
if config.cnn.img_type == "greyscale":
    config.cnn.img_shape = (input_size, input_size, 1)
else:
    config.cnn.img_shape = (input_size, input_size, 3)

config.cnn.input_shape = (input_size, input_size, 3)

st.markdown(
    """
    ### Train    
    """
)

config.training.batch_size = st.number_input("Batch size", value=config.training.batch_size, min_value=8, step=8)
config.training.max_epochs = st.number_input("Maximum number of epochs", value=config.training.max_epochs)
config.training.alr_epochs = st.number_input("Active learning rate: number of epochs to monitor before reducing learning rate", value=config.training.alr_epochs)
config.training.alr_drops = st.number_input("Active learning rate: number of times to reduce learning rate before stopping training", value=config.training.alr_drops)
config.training.monitor_val_loss = st.checkbox("Monitor validation loss instead of training", value=config.training.monitor_val_loss)
config.training.use_class_weights = st.checkbox("Use class weights (recommended)", value=config.training.use_class_weights)
config.training.use_augmentation = st.checkbox("Use augmentation (recommended)", value=config.training.use_augmentation)

st.markdown(
    """
    ### Save    
    """
)
default_output_path = Path(config.dataset.source).parent / "models"
config.output.output_dir = st.text_input("Output directory", value=str(default_output_path))
config.output.save_model = st.checkbox("Save model", value=config.output.save_model)
config.output.save_mislabeled = st.checkbox("Save mislabeled images", value=config.output.save_mislabeled)

st.session_state["config"] = config

run = st.form_submit_button("Run")

if run:
    train_model_task.delay(config.dumps())
    st.write("Training job has been added to queue. Check queue page for status.")

# event = st_file_browser(
#     "F:\\rx",
#     key="A",
# )
# st.write(event)

st.markdown("### Config")
st.markdown("```\n" + config.dumps() + "\n```")