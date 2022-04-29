import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.lines as lines
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
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
        count = cm.sum(axis=1)[:, np.newaxis]
        count[count == 0] = 1
        cm = cm.astype('float') / count
        cm = np.nan_to_num(cm)
        cm = np.round(cm * 100).astype(int)
    # Plot confusion matrix
    plt.figure(facecolor="white", figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # Add axes cls
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


def get_text_width(txt):
    f = plt.figure()
    r = f.canvas.get_renderer()
    t = plt.text(0.5, 0.5, txt)
    plt.tight_layout()
    bb = t.get_window_extent(renderer=r)
    width = bb.width
    plt.close('all')
    return width


def plot_confusion_accuracy_matrix(y_true,
                                   y_pred,
                                   cls_labels,
                                   normalise=True,
                                   title='Confusion matrix',
                                   cmap=plt.cm.Blues,
                                   figsize=None,
                                   style='grid5',
                                   show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    mult = 3
    # If figsize is None, estimate the plot size
    max_word = cls_labels[np.argmax([len(f"{lab}") for lab in cls_labels])]
    txt_width = get_text_width(max_word) / 40
    # print(txt_width)
    sz = len(cls_labels) / mult + txt_width + 1.5
    sz2 = len(cls_labels) / mult + 1.5
    if figsize is None:
        figsize = (sz, sz)
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
    # Create combined plots
    f, ax = plt.subplots(2, 2,
                         gridspec_kw={'width_ratios': [sz2 - 1.5, 1],
                                      'height_ratios': [1, sz2 - 1.5],
                                      'wspace': 0,
                                      'hspace': 0},
                         figsize=figsize)
    # Add counts to class cls
    cls_labels = ['{} ({})'.format(cls_labels[i], s[i]) for i in range(len(cls_labels))]
    thresh = cm.max() / 2.
    # Axes
    ax_cm = ax[1, 0]
    ax_right = ax[1, 1]
    ax_top = ax[0, 0]
    ax_unused = ax[0, 1]
    ax_unused.axis('off')
    # Axes cls
    tick_marks = np.arange(len(cls_labels))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(cls_labels, rotation=90)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(cls_labels)
    # ax_cm.set_xticks(tick_marks)
    # ax_cm.set_xticklabels(cls_labels)
    ax_cm.set_xlim(-0.5, len(cls_labels) - 0.5)
    ax_cm.set_ylim(-0.5, len(cls_labels) - 0.5)
    ax_cm.invert_yaxis()
    # Bar plots
    remove_frame(ax_top)
    ax_top.bar(tick_marks, p, width=0.8, color=cmap(p), edgecolor=(0, 0, 0, 0.6))
    ax_top.set_xlim((-0.5, len(tick_marks) - 0.5))
    for i, v in enumerate(p):
        if np.mean(cmap(v)[:-1]) < 0.5:
            clr = 'white'
        else:
            clr = 'black'
        ax_top.text(i, 0.15, '{:.1f}'.format(v * 100), color=clr, ha='center', rotation=90, alpha=0.7)
    remove_frame(ax_right)
    ax_right.barh(tick_marks, r, height=0.8, color=cmap(r), edgecolor=(0, 0, 0, 0.6))
    ax_right.set_ylim((-0.5, len(tick_marks) - 0.5))
    for i, v in enumerate(r):
        if np.mean(cmap(v)[:-1]) < 0.5:
            clr = 'white'
        else:
            clr = 'black'
        ax_right.text(0.05, i, '{:.1f}'.format(v * 100), color=clr, va='center', alpha=0.7)
    ax_right.invert_yaxis()
    # Confusion matrix
    patches = []
    colors = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        patches.append(pch.Rectangle((j - 0.5, i - 0.5), 1, 1))
        colors.append(cm[i, j] / 100)
        # if cm[i, j] != 0 or style == 'checker':
        ax_cm.text(j, i + 0.25, cm[i, j],
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")
    patcol = PatchCollection(patches, alpha=1, cmap=cmap)
    patcol.set_array(np.array(colors))
    ax_cm.add_collection(patcol)
    # Grid
    if style == 'grid':
        for i in range(len(cls_labels) - 1):
            line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(cls_labels) + 0.5], color=(0, 0, 0, 0.05))
            line2 = lines.Line2D([-0.5, len(cls_labels) + 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.05))
            ax_cm.add_line(line1)
            ax_cm.add_line(line2)
    if style == 'grid5':
        for i in range(len(cls_labels) - 1):
            if i % 5 == 4:
                line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(cls_labels) + 0.5], color=(0, 0, 0, 0.2))
                line2 = lines.Line2D([-0.5, len(cls_labels) + 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.2))
                ax_cm.add_line(line1)
                ax_cm.add_line(line2)
    # Labels
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')
    ax_right.set_ylabel('Recall {:.1f}%'.format(np.mean(r) * 100))
    ax_right.yaxis.set_label_position('right')
    ax_top.set_xlabel('Precision {:.1f}%'.format(np.mean(p) * 100))
    ax_top.xaxis.set_label_position('top')
    ax_top.set_title('Overall accuracy {:.1f}%'.format(accuracy_score(y_true, y_pred) * 100))
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


def plot_comparison_matrix(y_true,
                           y_pred,
                           true_cls_labels,
                           pred_cls_labels,
                           normalise=True,
                           title='Comparison matrix',
                           cmap=plt.cm.Blues,
                           figsize=None,
                           style='grid5',
                           show=False
                           ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # If figsize is None, estimate the plot size
    if figsize is None:
        figsize = (len(pred_cls_labels) / 2.75 + 2, len(true_cls_labels) / 2.75 + 2)

    # Calculate confusion matrix
    max_labels = np.max((len(true_cls_labels), len(pred_cls_labels)))
    cm = confusion_matrix(y_true=y_true,
                          y_pred=y_pred,
                          labels=range(max_labels))
    cm = cm[0:len(true_cls_labels), 0:len(pred_cls_labels)]
    support = np.sum(cm, axis=1)
    # Normalise the values
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm * 100).astype(int)
    # Create combined plots
    # f, ax = plt.subplots(2, 2,
    #                      gridspec_kw={'width_ratios': [6, 1],
    #                                   'height_ratios': [1, 6],
    #                                   'wspace': 0,
    #                                   'hspace': 0},
    #                      figsize=figsize)
    # Add counts to class cls

    true_cls_labels = ['{} ({})'.format(true_cls_labels[i], support[i]) for i in range(len(true_cls_labels))]
    thresh = cm.max() / 2.

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Axes cls
    true_tick_marks = np.arange(len(true_cls_labels))
    pred_tick_marks = np.arange(len(pred_cls_labels))

    ax.set_yticks(true_tick_marks)
    ax.set_yticklabels(true_cls_labels)
    ax.set_ylim(-0.5, len(true_cls_labels) - 0.5)

    ax.set_xticks(pred_tick_marks)
    ax.set_xticklabels(pred_cls_labels, rotation=90)
    ax.set_xlim(-0.5, len(pred_cls_labels) - 0.5)

    ax.invert_yaxis()

    # Confusion matrix
    patches = []
    colors = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        patches.append(pch.Rectangle((j - 0.5, i - 0.5), 1, 1))
        colors.append(cm[i, j] / 100)
        # if cm[i, j] != 0 or style == 'checker':
        ax.text(j, i + 0.25, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    patcol = PatchCollection(patches, alpha=1, cmap=cmap)
    patcol.set_array(np.array(colors))
    ax.add_collection(patcol)

    # Grid
    if style == 'grid':
        for i in range(len(pred_cls_labels) - 1):
            line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(true_cls_labels) + 0.5], color=(0, 0, 0, 0.05))
            ax.add_line(line1)
        for i in range(len(true_cls_labels) - 1):
            line2 = lines.Line2D([-0.5, len(pred_cls_labels) + 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.05))
            ax.add_line(line2)

    if style == 'grid5':
        for i in range(len(pred_cls_labels) - 1):
            if i % 5 == 4:
                line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(true_cls_labels) + 0.5], color=(0, 0, 0, 0.2))
                line2 = lines.Line2D([-0.5, len(true_cls_labels) + 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.2))
                ax.add_line(line1)
                ax.add_line(line2)
        for i in range(len(true_cls_labels) - 1):
            if i % 5 == 4:
                line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(pred_cls_labels) + 0.5], color=(0, 0, 0, 0.2))
                line2 = lines.Line2D([-0.5, len(pred_cls_labels) + 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.2))
                ax.add_line(line1)
                ax.add_line(line2)
    # Labels
    ax.set_ylabel('True cls')
    ax.set_xlabel('Predicted cls')
    plt.tight_layout()
    if show is True:
        plt.show()

#
# if __name__ == "__main__":
#     import string
#     import random
#
#     def get_random_string(length):
#         # choose from all lowercase letter
#         letters = string.ascii_lowercase
#         result_str = ''.join(random.choice(letters) for i in range(length))
#         # print("Random string of length", length, "is:", result_str)
#         return result_str
#
#     offset = 0
#     for i in [3,4,30,50]:
#         y_true = np.arange(i+offset)
#         y_pred = np.arange(i+offset)
#         cls_labels = [get_random_string(np.random.randint(3,30)) for j in range(i+offset)]
#         cls_labels[0] = "DIAT-Chaetoceros"
#         cls_labels[1] = "DIAT-Other"
#         cls_labels[2] = "DIAT-Pseudo-nitzschia"
#         # cls_labels[0] = "DIAT-"
#         # cls_labels[1] = "DIAT-"
#         # cls_labels[2] = "DIAT-"
#         if len(cls_labels) > 3:
#             cls_labels[3] = "DIAT-Pseudo-nitzschia"
#         print(cls_labels)
#         plot_confusion_accuracy_matrix(y_true,
#                                        y_pred,
#                                        cls_labels,
#                                        normalise=True,
#                                        title='Confusion matrix',
#                                        cmap=plt.cm.Blues,
#                                        figsize=None,
#                                        style='grid5',
#                                        show=False)
#         plt.show()
