import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.lines as lines
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from matplotlib.collections import PatchCollection


def plot_confusion_matrix(y_true,
                          y_pred,
                          cls_labels,
                          normalise=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=None,
                          style='checker',
                          show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # If figsize is None, estimate the plot size
    if figsize is None:
        figsize = (len(cls_labels) / 3, len(cls_labels) / 3)
    # Calculate precision, recall, etc
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(cls_labels)))
    # Calculate confusion matrix
    cm = confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          labels=range(len(cls_labels)))
    # Normalise the values
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        cm = np.round(cm * 100).astype(int)
    # Plot confusion matrix
    plt.figure(facecolor="white", figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # Add axes labels
    tick_marks = np.arange(len(cls_labels))
    plt.xticks(tick_marks, cls_labels, rotation=90)
    plt.yticks(tick_marks, cls_labels)
    # Add grid
    if style == 'grid':
        ax = plt.gca()
        minor_tick_marks = tick_marks[:-1] + 0.5
        ax.set_xticks(minor_tick_marks, minor=True)
        ax.set_yticks(minor_tick_marks, minor=True)
        plt.grid(which='minor')
    # Add percentages in boxes
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0 or style == 'checker':
            plt.text(j, i + 0.25, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if show is True:
        plt.show()


def plot_confusion_accuracy_matrix(y_true,
                                   y_pred,
                                   cls_labels,
                                   normalise=True,
                                   title='Confusion matrix',
                                   cmap=plt.cm.Blues,
                                   figsize=None,
                                   style='checker',
                                   show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # If figsize is None, estimate the plot size
    if figsize is None:
        figsize = (len(cls_labels) / 2.75 + 2, len(cls_labels) / 2.75 + 2)
    # Calculate precision, recall, etc
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(cls_labels)))
    print(p)
    print(r)
    # Calculate confusion matrix
    cm = confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          labels=range(len(cls_labels)))
    # Normalise the values
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm * 100).astype(int)
    # Create combined plots
    f, ax = plt.subplots(2, 2,
                         gridspec_kw={'width_ratios': [6, 1],
                                      'height_ratios': [1, 6],
                                      'wspace': 0,
                                      'hspace': 0},
                         figsize=figsize)
    # Axes
    ax_cm = ax[1, 0]
    ax_left = ax[1, 1]
    ax_bot = ax[0, 0]
    ax_unused = ax[0, 1]
    ax_unused.axis('off')
    # Axes labels
    tick_marks = np.arange(len(cls_labels))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(cls_labels, rotation=90)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(cls_labels)
    ax_cm.set_xticks(tick_marks, cls_labels)
    ax_cm.set_xlim(-0.5, len(cls_labels)-0.5)
    ax_cm.set_ylim(-0.5, len(cls_labels) - 0.5)
    ax_cm.invert_yaxis()
    # Bar plots
    remove_frame(ax_bot)
    ax_bot.bar(tick_marks, p, width=0.8, color=cmap(p), edgecolor=(0,0,0,0.6))
    ax_bot.set_xlim((-0.5, len(tick_marks) - 0.5))
    remove_frame(ax_left)
    ax_left.barh(tick_marks, r, height=0.8, color=cmap(r), edgecolor=(0,0,0,0.6))
    ax_left.set_ylim((-0.5, len(tick_marks) - 0.5))
    ax_left.invert_yaxis()
    # Confusion matrix
    thresh = cm.max() / 2.
    patches = []
    colors = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        patches.append(pch.Rectangle((j-0.5, i-0.5), 1, 1))
        colors.append(cm[i, j] / 100)
        if cm[i, j] != 0 or style == 'checker':
            ax_cm.text(j, i + 0.25, cm[i, j],
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
    p = PatchCollection(patches, alpha=1, cmap=cmap)
    p.set_array(np.array(colors))
    ax_cm.add_collection(p)
    # Grid
    if style == 'grid':
        for i in range(len(cls_labels)-1):
            line1 = lines.Line2D([i+0.5,i+0.5],[-0.5,len(cls_labels)+0.5], color=(0,0,0,0.05))
            line2 = lines.Line2D([-0.5, len(cls_labels) + 0.5],[i + 0.5, i + 0.5], color=(0, 0, 0, 0.05))
            ax_cm.add_line(line1)
            ax_cm.add_line(line2)
    # Labels
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')
    ax_left.set_ylabel('Recall')
    ax_left.yaxis.set_label_position('right')
    ax_bot.set_xlabel('Precision')
    ax_bot.xaxis.set_label_position('top')
    ax_cm.set_zorder(100)
    plt.tight_layout()
    if show is True:
        plt.show()


def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
