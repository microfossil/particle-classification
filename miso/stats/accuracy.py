import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def plot_precision_recall(y_true,
                          y_pred,
                          cls_labels,
                          fig_size=(6, 4),
                          rotate_labels=90,
                          show=False):
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(cls_labels)))
    plt.figure(facecolor="white", figsize=fig_size)
    ax = plt.subplot(111)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    w = 0.2
    x = np.arange(len(p))
    class_precision = p * 100
    class_recall = r * 100
    ax = plt.gca()
    h1 = ax.bar(x - w, class_recall, width=2 * w, align='center')
    h2 = ax.bar(x + w, class_precision, width=2 * w, align='center')
    h3 = ax.plot(x, f1*100, 'k.')
    plt.xticks(x, cls_labels, rotation=rotate_labels)
    plt.ylim(0, 100)
    plt.xlim(x[0]-0.5, x[-1]+0.5)
    plt.legend((h1[0], h2[0], h3[0]),
               ['recall: {0:.1f}%'.format(np.mean(r) * 100),
                'precision: {0:.1f}%'.format(100 * np.mean(p)),
                'f1: {0:.1f}%'.format(100 * np.mean(f1))],
               loc=(0, 1.04),
               ncol=3)
    plt.xlabel('Accuracy {0:.1f}%'.format(accuracy_score(y_true, y_pred)*100))
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    if show:
        plt.show()
