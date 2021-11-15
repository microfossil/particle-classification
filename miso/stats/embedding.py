"""
Set of methods to make some common plots for the CNN outputs
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial.distance import cdist
import scipy.ndimage as nd


def plot_embedding(X, y, num_classes, labels=None, title=None, indices=None, alpha=1.0, figsize=(8,8)):
    """
    Scatter plot of classes and their 2D embedding
    :param X: The 2D co-ordinates of each class value
    :param y: The classes
    :param num_classes: Total number of classes
    :param labels: Names of each class (if None, y values is used)
    :param title: Title of the plot
    :param indices: Extra label to append to each class point, e.g. the index of the image
    :param alpha: Transparency of the class label (0 - fully transparent, 1 - fully opaque)
    :param figsize: Size of the figure
    :return:
    """
    if labels is None:
        labels = ["{}".format(i) for i in range(num_classes)]
    # print(labels)
    # TODO: This can be simplified
    target_values = np.array(range(num_classes))
    colors = target_values / num_classes
    plt.figure(figsize=figsize)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    plt.axis([x_min[0], x_max[0], x_min[1], x_max[1]])
    points = [[] for k in range(len(target_values))]
    for i in range(X.shape[0]):
        for j in range(len(target_values)):
            if y[i] == target_values[j]:
                lbl = "{}".format(labels[y[i]])
                if indices is not None:
                    lbl += " {}".format(indices[i])
                plt.text(X[i, 0], X[i, 1], lbl,
                         color=plt.cm.tab20(colors[j]),
                         fontdict={'weight': 'bold', 'size': 9},alpha=alpha)
                points[j].append([X[i, 0], X[i, 1]])
                break
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.gca().axis('off')


def plot_embedding_with_images(X, y, num_classes, images, face_colour="black", class_border=False, mask_level=(0.3, 0.7), scale_adj=1.0, figsize=None):
    """
    Scatter plot of classes and their 2D embedding. Instead of class name, the actual image is used.
    The dark parts of the images will be masked (used mainly for foraminifera images)
    :param X: The 2D co-ordinates of each class value
    :param y: The classes
    :param num_classes: Total number of classes
    :param images: The images corresponding to each class entry
    :param face_colour: Colour of the figure background
    :param class_border: Make a border with the colour of the class (not implemented)
    :param mask_level: Two values [0] is mask threshold, [1] is intensity above which everything is opaque
    :param scale_adj:
    :param figsize: Size of the figure. If None, it is scaled automatically (recommended)
    :return:
    """
    # Normalise the co-ordinates
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Calculate the scaling amount
    scale = images.shape[1] // 64 / scale_adj
    # Create the figure
    if figsize is None:
        figsize = (20*scale, 20*scale)
    plt.figure(figsize=figsize, dpi=50, facecolor=face_colour, edgecolor=face_colour)
    plt.gca().set_facecolor(face_colour)
    # Plot
    clrs = plt.cm.get_cmap('tab20')
    for j in range(X.shape[0]):
        img = np.squeeze(images[j,:,:,0])
        # Create a mask with a soft border
        # mask_level[0] is the threshold
        # mask_level[1] is the intensity value above which everything is fully opaque
        mask = img > mask_level[0]
        mask = nd.morphology.binary_fill_holes(mask)
        # if class_border:
        #     clr = clrs(y/num_classes)
        #     cmask = np.stack((mask*clr[0], mask*clr[1], mask*clr[2], np.ones(mask.shape)), axis=-1)
        th = img / mask_level[1]
        th[th > 1] = 1
        th[mask == True] = 1
        img = np.stack((img, img, img, th), axis=-1)
        ab = AnnotationBbox(OffsetImage(img), (0.03 + X[j, 0]*0.94, 0.03 + X[j, 1] * 0.94), xycoords="axes fraction", frameon=False)
        plt.gca().add_artist(ab)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


def plot_embedding_with_colour_images(X, images, face_colour="black", scale_adj=1.0, figsize=(10,10)):
    # Normalise the co-ordinates
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Calculate the scaling amount
    scale = images.shape[1] // 64 / scale_adj
    # Create the figure
    if figsize is None:
        figsize = (20*scale, 20*scale)
    plt.figure(figsize=figsize, dpi=50, facecolor=face_colour, edgecolor=face_colour)
    plt.gca().set_facecolor(face_colour)
    # Plot
    for j in range(X.shape[0]):
        img = images[j,:,:,:]
        ab = AnnotationBbox(OffsetImage(img), (0.03 + X[j, 0]*0.94, 0.03 + X[j, 1] * 0.94), xycoords="axes fraction", frameon=False)
        plt.gca().add_artist(ab)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


def plot_embedding_with_images_on_grid(X, y, num_classes, images, width=128, mask_level=0.7, scale_adj=1.0):
    """
    2D image scatter plots, but with the images in a grid.
    Requires lapjv which can be installed with 'pip install lapjv' BUT you need a C compiler installed as well
    :param X:
    :param y:
    :param num_classes:
    :param images:
    :param width:
    :param mask_level:
    :param scale_adj:
    :return:
    """
    grid_width = int(np.min((np.floor(np.sqrt(len(y))),20)))
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, grid_width), np.linspace(0, 1, grid_width))).reshape(-1, 2)
    X = X[0:grid_width*grid_width,:]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    cost_matrix = cdist(grid, X, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    plot_embedding_with_images(grid_jv, y, num_classes, images, face_colour="black", class_border=False, mask_level=mask_level, scale_adj=scale_adj)
