from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.utils.extmath import weighted_mode
from sklearn.preprocessing import normalize
import os

from tqdm import tqdm


def find_and_save_mislabelled(images,
                              vectors,
                              cls,
                              cls_labels,
                              image_names,
                              output_dir,
                              num_neighbours=11):
    """
    Uses nearest neighbour to classify the vectors and checks this matches the actual classifications.
    If not, prints the most similar other images. Also writes to a csv file with this list.
    Args:
        images: Images (for use in plots)
        vectors: Vectors for each image
        cls: Class label for each image
        cls_labels: List of class cls in order [class_0, class_1, ..., class_N]
        image_names: List of ids for each image, e.g. their filenames
        output_dir: Directory to save the results
        num_neighbours: Number of neighbours to use for the kNN classification
    Returns: None
    """
    # Normalise vectors
    vectors = normalize(vectors, axis=1)
    # Nearest neighbours fit
    neigh = KNeighborsClassifier(n_neighbors=num_neighbours + 1, algorithm='brute')
    neigh.fit(vectors, cls)

    # Find the k nearest vectors
    # The first will be the same vector as passed in so ignore it
    knn = neigh.kneighbors(vectors, n_neighbors=num_neighbours + 1)
    distance = knn[0][:, 1:]
    idx = knn[1][:, 1:]
    # Get the predicted class from knn (ignoring the same vector)
    knn_cls = cls[idx]
    wm = weighted_mode(knn_cls, np.max(distance) - distance, axis=1)
    pred_cls = wm[0].astype(int).flatten()
    pred_w = wm[1]

    # Get the index of images with a different k-NN prediction to their label
    diff_idx = np.where(cls != pred_cls)[0]

    # Write a CSV of the mislabelled
    with open(os.path.join(output_dir, "mislabeled.csv"), 'w+') as f:
        f.write("filename, label, predicted_label\n")
        for i in diff_idx:
            f.write("{},{},{}\n".format(image_names[i], cls_labels[cls[i]], cls_labels[pred_cls[i]]))

    if num_neighbours == 1:
        nx = 2
        ny = 1
    elif num_neighbours == 2:
        nx = 3
        ny = 1
    elif num_neighbours == 3:
        nx = 2
        ny = 2
    elif num_neighbours == 5:
        nx = 3
        ny = 2
    else:
        nx = 4
        ny = int(np.ceil((num_neighbours + 1) / 4))

    # Plot each image
    for im_idx in tqdm(diff_idx):
        fig, axes = plt.subplots(nrows=ny, ncols=nx, figsize=(2 * nx, 2 * ny + 1),
                                 gridspec_kw={"top": (2 * ny + 0.5) / (2 * ny + 1),
                                              "left": 0.05,
                                              "right": 0.95,
                                              "bottom": 0.5 / (2 * ny + 1)})
        plt.set_cmap('gray')
        fig.set_facecolor("white")
        for i, ax in enumerate(axes.flat):
            if i == 0:
                image = images[im_idx].astype(np.uint8)
                if images.shape[3] == 1:
                    ax.imshow(image[:, :, 0])
                else:
                    ax.imshow(image)
                if cls_labels is None:
                    xlabel = "{0}\n({1})?".format(cls[im_idx], pred_cls[im_idx])
                else:
                    xlabel = "{0}\n{1}?".format(cls_labels[cls[im_idx]], cls_labels[pred_cls[im_idx]])
                ax.set_xlabel(xlabel)
            else:
                kidx = idx[im_idx, i - 1]
                image = images[kidx].astype(np.uint8)
                if images.shape[3] == 1:
                    ax.imshow(image[:, :, 0])
                else:
                    ax.imshow(image)
                if cls_labels is None:
                    xlabel = "{0}".format(cls[kidx])
                else:
                    xlabel = "{0}".format(cls_labels[cls[kidx]])
                ax.set_xlabel(xlabel)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        # Tight layout looks nicer
        fig.tight_layout()
        plt.suptitle(image_names[im_idx])
        # print("- actual: {} predicted: {} - {}".format(cls_labels[cls[im_idx]], cls_labels[pred_cls[im_idx]], image_names[im_idx]))

        filename = str(Path(image_names[im_idx]).stem) + ".png"
        path = Path(output_dir) / "mislabeled" / cls_labels[cls[im_idx]]
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / filename)
        plt.clf()
        plt.close('all')
