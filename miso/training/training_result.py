"""
Results of training
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class TrainingResult:
    def __init__(self,
                 model_params,
                 history,
                 y_true,
                 y_pred,
                 y_prob,
                 cls_labels,
                 training_time,
                 inference_time):

        # Model configuration
        self.model_params = model_params
        self.cls_labels = cls_labels

        # Training statistics
        self.training_time = training_time
        self.inference_time = inference_time

        # Overall accuracy
        self.accuracy = accuracy_score(y_true, y_pred)

        # Training history
        self.epochs = history.epoch
        self.loss = history.history['loss']
        self.acc = history.history.get('acc') or history.history.get('accuracy')
        if 'val_loss' in history.history.keys():
            self.val_loss = history.history['val_loss']
            self.val_acc = history.history.get('val_acc') or history.history.get('val_accuracy')
        else:
            self.val_loss = []
            self.val_acc = []

        # Accuracy metrics
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(self.cls_labels)))
        self.recall = r
        self.precision = p
        self.f1_score = f1
        self.support = s
        self.mean_precision = np.mean(self.precision)
        self.mean_recall = np.mean(self.recall)
        self.mean_f1_score = np.mean(self.f1_score)

        # Save predictions (for later analysis)
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

