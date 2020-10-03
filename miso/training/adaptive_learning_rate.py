import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from miso.training.rolling_buffer import RollingBuffer
import math
import time


def graph_to_console(epoch, batch, acc, loss, val_acc, val_loss, lr_prob, lr_prob_active, time_difference):
    acc_i = round(acc * 100)
    val_acc_i = round(val_acc * 100)
    lr_prob_i = round(lr_prob * 100)

    for j in range(101):
        if j == acc_i:
            print('#', end="")
        # elif j == trainsdi:
        #     print('@', end="")
        elif j == val_acc_i:
            print('*', end="")
        elif j == lr_prob_i and j != 100 and lr_prob_active:
            print('+', end="")
        elif j % 10 == 0:
            print('|', end="")
        else:
            print(' ', end="")
    msg = " Epoch:{0:>6.1f} Batch:{1:>6} #Train: {2:>5.1f}% ({3:>6.4f})," \
          " *Val: {4:>5.1f}% ({5:>6.4f}) !Prob{6:>5.1f} Time: {7:>5.2f}s"
    print(msg.format(epoch, batch, acc * 100, loss, val_acc * 100, val_loss, lr_prob, time_difference))


class AdaptiveLearningRateScheduler(Callback):
    """
    Adaptive learning rate scheduler

    Decreases learning rate by a certain factor each time it is no longer improving
    """

    def __init__(self, drop_rate=0.5, nb_drops=4, nb_epochs=10, verbose=1, monitor='loss'):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.monitor = monitor
        self.drop_rate = drop_rate
        self.nb_drops = nb_drops
        self.nb_epochs = nb_epochs
        self.verbose = verbose
        self.current_epoch = 0
        self.current_batch = 0
        self.drop_count = 0
        self.buffer = None
        self.previous_time = None
        self.finished = False

    def on_train_begin(self, logs=None):
        if 'batch_size' in self.params and self.params['batch_size'] is not None:
            batch_size = self.params['batch_size']
            samples = self.params['samples']
            self.buffer = RollingBuffer(math.ceil(samples * self.nb_epochs / batch_size))
        else:
            self.buffer = RollingBuffer(self.params['steps'] * self.nb_epochs)
        self.previous_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        acc = logs.get("acc") or logs.get('accuracy') or logs.get('iou_score') or logs.get('cosine_proximity')
        val_loss = logs.get("val_loss") or 0
        val_acc = logs.get("val_acc") or logs.get('val_accuracy') or logs.get('val_iou_score') or logs.get(
            'val_cosine_proximity') or 0

        if 'cosine_proximity' in logs:
            acc += 1
            val_acc += 1

        current_time = time.time()
        if self.previous_time is None:
            time_difference = 0
        else:
            time_difference = current_time - self.previous_time
        self.previous_time = current_time
        if val_acc is not None:
            print("\r", end="")
            graph_to_console(self.current_epoch, self.current_batch,
                             acc, loss, val_acc, val_loss,
                             self.buffer.slope_probability_less_than(0), self.buffer.full(),
                             time_difference)
        if self.finished is True:
            self.model.stop_training = True
            print("@Training finished".format(self.model.optimizer.lr))

    def on_batch_end(self, batch, logs=None):
        monitor_value = logs.get(self.monitor)
        self.buffer.append(monitor_value)
        self.current_batch += 1
        # print("\r{}".format(self.current_batch), end="")

        if self.current_batch > self.buffer.length() * 3 and self.buffer.full() and self.finished is False:
            if self.buffer.slope_probability_less_than(0) < 0.50:
                lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = lr * self.drop_rate
                K.set_value(self.model.optimizer.lr, new_lr)
                self.buffer.clear()
                self.drop_count += 1

                if self.drop_count == self.nb_drops:
                    self.finished = True
                    return

                if self.verbose == 1:
                    print("@Learning rate dropped ({}/{})".format(self.drop_count, self.nb_drops))
