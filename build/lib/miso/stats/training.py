"""
Plots of training and accuracy
"""
import numpy as np
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt


def plot_loss_vs_epochs(history: History, figsize=(8, 4)):
    epochs = history.epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=figsize)
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.grid()
    plt.title("Loss")
    plt.legend(("Training", "Validation"))
    plt.xlim(left=0)
    plt.ylim(bottom=0)


def plot_accuracy_vs_epochs(history: History, metric='acc', figsize=(8, 4)):
    epochs = history.epoch
    acc = np.asarray(history.history[metric])
    val_acc = np.asarray(history.history['val_' + metric])
    plt.figure(figsize=figsize)
    plt.plot(epochs, acc * 100)
    plt.plot(epochs, val_acc * 100)
    plt.grid()
    plt.title("Accuracy")
    plt.legend(("Training", "Validation"))
    plt.xlim(left=0)
    plt.ylim(top=100)


