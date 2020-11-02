import argparse
from miso.training.trainer import train_image_classification_model
from miso.training.parameters import MisoParameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN to classify images")
    parser.add_argument("-i", "--input", required=True, help="Directory of images, URL link to zipped directory of images, or ParticleTrieur project file")
    parser.add_argument("-o", "--output", required=True, help="Output directory to store training results")
    parser.add_argument("-t", "--type", required=True, help="Type of CNN: resnet_tl, resnet_cyclic_tl, base_cyclic, resnet_cyclic, resnet18, resnet50, vgg16, vgg19")

    parser.add_argument("-f", "--filters", type=int, default=4, help="Number of filters in the first convolutional block")

    parser.add_argument("--min_count", type=int, default=10, help="Minimum number of images in a class for it to be included")
    parser.add_argument("--map_others", action='store_true', help="Classes with not enough images will be put into 'others' class (so long as the total is also greater than min_count")
    args = parser.parse_args()

    tp = MisoParameters()
    tp.source = args.input
    tp.output_dir = args.output
    tp.cnn_type = args.type
    tp.filters = args.filters
    tp.min_count = args.min_count
    tp.map_others = args.map_others

    train_image_classification_model(tp)
    # print("This is the name of the script: ", sys.argv[0])
    # print("Number of arguments: ", len(sys.argv))
    # print("The arguments are: ", str(sys.argv))
    #
    # params = default_params()
    # print(params)
    # key = None
    # for arg in sys.argv[1:]:
    #     print(arg)
    #     if arg.startswith('--'):
    #         key = arg[2:]
    #         print(key)
    #     else:
    #         try:
    #             val = int(arg)
    #             print("int")
    #         except ValueError:
    #             try:
    #                 val = float(arg)
    #                 print("float")
    #             except:
    #                 val = arg
    #                 print("string")
    #         params[key] = val
    # print(params)
    # train_image_classification_model(params)

