import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min


def plot_most_representative(images, vectors, cls, cls_labels, output_dir):
    # Normalise vectors
    vectors = normalize(vectors, axis=1)
    # Find centroids
    clf = NearestCentroid()
    clf.fit(vectors, cls)
    # Get nearest
    im_idx, dist = pairwise_distances_argmin_min(clf.centroids_, vectors)

    for i, idx in enumerate(im_idx):
        plt.clf()
        plt.imshow(images[idx])
        plt.axis('off')
        plt.title('{}  -  {}'.format(i, cls_labels[i]))
        plt.tight_layout()
        plt.show()
        output_file = os.path.join(output_dir, 'archetypes', '{}_{}.png'.format(i, cls_labels[i]))
        plt.imsave(output_file)

