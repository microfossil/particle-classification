import sys

"""
The parameters:

Datasource:
- data_source
- data_min_count
- data_load_to_memory

Input:
- img_height
- img_width
- img_channels

Network:
- cnn_blocks
- cnn_dropout
- cnn_dense

Training:
- training_optimiser = Adam
- training_lr = 1e-3
- training_batch_size = 64
- training_max_epochs = 300
- training_alr_epochs = 40
- training_alr_drops = 4
- training_alr_factor = 0.5

Augmentation
- aug_method = "tensorflow" or "keras"
- aug_rotation
- aug_gain
- aug_zoom
- aug_gamma
- aug_bias
- aug_noise

pip install -U git+http://systrifor:Cerege2018@bitbucket.org/projetfirst/particleclassification.git
"""
from miso.training.model_params import default_params
from .training.model_trainer import train_image_classification_model

if __name__ == "__main__":
    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: ", str(sys.argv))

    params = default_params()
    print(params)
    key = None
    for arg in sys.argv[1:]:
        print(arg)
        if arg.startswith('--'):
            key = arg[2:]
            print(key)
        else:
            try:
                val = int(arg)
                print("int")
            except ValueError:
                try:
                    val = float(arg)
                    print("float")
                except:
                    val = arg
                    print("string")
            params[key] = val
    print(params)
    train_image_classification_model(params)

