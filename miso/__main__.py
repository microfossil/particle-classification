import click

# Cant have name same as cli method name
from miso.inference.classify import classify_folder as miso_classify_folder, segment_folder
from miso.inference.classify import segment_folder as miso_segment_folder


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='Path to the model information.')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Path to the directory containing images.')
@click.option('--output', '-o', type=click.Path(), required=True, help='Path where the output CSV will be saved.')
@click.option('--batch_size', '-b', type=int, default=32, show_default=True, help='Batch size for processing images.')
@click.option('--sample', '-s', type=str, default='unknown', show_default=True, help='Default sample name')
@click.option('--unsure_threshold', '-u', type=float, default=0.0, show_default=True, help='Threshold below which predictions are considered unsure.')
def classify_folder(model, input, output, batch_size, sample, unsure_threshold):
    """
    Classify images in a folder and output the results to a CSV file.
    """
    miso_classify_folder(model, input, output, batch_size, sample, unsure_threshold)


@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='Path to the model information.')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Path to the folder containing images.')
@click.option('--output', '-o', type=click.Path(), required=True, help='Path where the morphology csv will be saved.')
@click.option('--batch_size', '-b', type=int, default=32, show_default=True, help='Batch size for processing images.')
@click.option('--sample', '-s', type=str, default='unknown', show_default=True, help='Default sample name')
@click.option('--threshold', '-t', type=float, default=0.5, show_default=True, help='Threshold for segmentation.')
@click.option('--save-contours', is_flag=True, default=False, help='Whether to save contours or not.')
def segment_folder(model, input, output, batch_size, sample, threshold, save_contours):
    """
    Segment images in a folder and output the results.
    """
    miso_segment_folder(model, input, output, batch_size, sample, threshold, save_contours)



if __name__ == "__main__":
    cli()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a CNN to classify images")
#     parser.add_argument("-i", "--input", required=True, help="Directory of images, URL link to zipped directory of images, or ParticleTrieur project file")
#     parser.add_argument("-o", "--output", required=True, help="Output directory to store training results")
#     parser.add_argument("-t", "--type", required=True, help="Type of CNN: resnet_tl, resnet_cyclic_tl, base_cyclic, resnet_cyclic, resnet18, resnet50, vgg16, vgg19")
#
#     parser.add_argument("-f", "--filters", type=int, default=4, help="Number of filters in the first convolutional block")
#
#     parser.add_argument("--min_count", type=int, default=10, help="Minimum number of images in a class for it to be included")
#     parser.add_argument("--map_others", action='store_true', help="Classes with not enough images will be put into 'others' class (so long as the total is also greater than min_count")
#     args = parser.parse_args()
#
#     tp = MisoParameters()
#     tp.source = args.input
#     tp.save_dir = args.output
#     tp.cnn_type = args.type
#     tp.filters = args.filters
#     tp.min_count = args.min_count
#     tp.map_others = args.map_others
#
#     train_image_classification_model(tp)
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

