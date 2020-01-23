import tflearn


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, acc_thresh):
        """
        Args:
            acc_thresh - if our accuracy > acc_thresh, terminate training.
        """
        self.acc_thresh = acc_thresh

    def on_epoch_end(self, training_state):
        """ """
        if training_state.val_acc is not None and training_state.val_acc > self.acc_thresh:
            print("Terminating training at the end of epoch", training_state.epoch)
            raise StopIteration

    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training,
        (or when training ends regardless). This would be a good time to store any additional
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)