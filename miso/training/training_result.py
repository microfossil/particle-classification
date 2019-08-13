import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class TrainingResult:

    def __init__(self, model_params, history, y_true, y_pred, y_prob, cls_labels, training_time):

        # Model configuration
        self.model_params = model_params
        self.cls_labels = cls_labels

        # Training statistics
        self.training_time = training_time

        # Overall accuracy
        self.accuracy = accuracy_score(y_true, y_pred)

        # Training history
        self.epochs = history.epoch
        self.loss = history.history['loss']
        self.acc = history.history['acc']
        self.val_loss = history.history['val_loss']
        self.val_acc = history.history['val_acc']

        # Class history
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
        self.recall = r
        self.precision = p
        self.f1_score = f1
        self.support = s

        # Test predictions (for later analysis
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def mean_precision(self):
        return np.mean(self.precision)

    def mean_recall(self):
        return np.mean(self.recall)

    def mean_f1_score(self):
        return np.mean(self.f1_score)

