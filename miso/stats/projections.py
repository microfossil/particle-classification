from sklearn import discriminant_analysis, manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np


def pca(X, nr_components=16, normalise_vectors=True):
    if normalise_vectors:
        X = normalize(X, axis=1)
    p = PCA(n_components=nr_components)
    p.fit(X)
    print(p.explained_variance_ratio_)
    print(np.sum(p.explained_variance_ratio_))
    return p.transform(X)


def lda(X, y, nr_components=2):
    """
    Linear discrimindant analysis
    :param X: Input vectors
    :param y: Input classes
    :param nr_components: Dimension of output co-ordinates
    :return: Output co-ordinates
    """
    print("Computing Linear Discriminant Analysis projection")
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
    return discriminant_analysis.LinearDiscriminantAnalysis(n_components=nr_components).fit_transform(X2, y)


def tsne(X, nr_components=2, perplexity=30, early_exaggeration=12):
    """
    t-SNE clustering
    :param X: Input vector
    :param nr_components: Dimension of output co-ordinates
    :param perplexity: t-SNE perplexity
    :param early_exaggeration: t-SNE early exaggeration
    :return: Output co-ordinates
    """
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=nr_components, init='random', random_state=0, perplexity=perplexity, early_exaggeration=early_exaggeration)
    return tsne.fit_transform(X)
