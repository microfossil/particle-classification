import click

from miso.inference.classify import classify_folder


@click.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help='Path to the model information XML.')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Path to the directory containing images.')
@click.option('--output', '-o', type=click.Path(), required=True, help='Path of the output CSV to be saved.')
@click.option('--batch_size', '-b', type=int, default=32, show_default=True, help='Batch size for processing images.')
@click.option('--in_samples/--no-in_samples', '-f', default=False, show_default=True, help='Set this flag if images are stored in subfolders, using the subfolder names as sample labels.')
@click.option('--sample', '-s', type=str, default='unknown', show_default=True, help='Default sample name if not using subfolders.')
@click.option('--unsure_threshold', 'u', type=float, default=0.0, show_default=True, help='Threshold below which predictions are considered unsure.')
def classify_folder(model, folder, output, batch_size, in_samples, sample, unsure_threshold):
    """
    Classify images in a folder and output the results to a CSV file.
    """
    classify_folder(model, folder, output, batch_size, in_samples, sample, unsure_threshold)