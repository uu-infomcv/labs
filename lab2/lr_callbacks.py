from keras.callbacks import LearningRateScheduler, Callback
import numpy as np
from math import exp
import keras
from keras import backend as K
import matplotlib.pyplot as plt

"""
--------------------------------------------------------------------------------------------------------
                S T A R T   O F  C L A S S    I N T E R N A L S T A T E H I S T O R Y
--------------------------------------------------------------------------------------------------------
[About]

    Class for returning the network's internal state parameter values per training phase epoch.
    Records the loss history, accuracy, learning rate and number of iterations. This class is
    to be used by other classes and functions when implementing custom LearningRateChedulers
    and loss functions that are dependable on the previous values or the functions include
    a adaptive methods.


[Arguments]

    None


"""

class InternalStateHistory(Callback):

    """
    --------------------------- F U N C T I O N   O N _ T R A I N _ B E G I N -------------------------
    """
    def on_train_begin(self, logs={}):

      # [1] initialise the logged losses to an empty list

      # [2] initialise the logged learning rates to an empty list

      # [4] initialise the logged accuraccy rates to an empty list

      # [5] initialise the current iterration number to zero

    """
    -------------------------------------- E N D   F U N C T I O N ------------------------------------
    """


    """
    ----------------------------- F U N C T I O N   O N _ E P O C H _ E N D ---------------------------
    """
    def on_epoch_end(self, batch, logs={}):

      # [1] append to the losses array the current loss by calling logs.get('loss')

      # [2] append to the lr values array the current lr by calling from keras backend (imported as K) the `get_value` function with `self.model.optimizer.lr` as argument.

      # [3] append the accuracy rate to the currosponding array similarly to how you obtained the loss.

      # [4] get the iteration number from self.model.optimizer - as you did for the lr
    """
    -------------------------------------- E N D   F U N C T I O N ------------------------------------
    """


"""
--------------------------------------------------------------------------------------------------------
                E N D   O F  C L A S S    I N T E R N A L S T A T E H I S T O R Y
--------------------------------------------------------------------------------------------------------
"""





"""
--------------------------------------------------------------------------------------------------------
                S T A R T   O F  C L A S S    S T E P _  D E C A Y
--------------------------------------------------------------------------------------------------------
[About]

    Class for decreasing the network's learning rate by a specific rate based on the upper and lower bound values given.


[Arguments]

    - min_lr: float defaulting to 1e-6, the minimum learning rate that the network can use (the lr in the final step).
    - max_lr: float defaulting to .1, the maximum learning rate that the network can use (the lr in the first step).
    - steps_per_epoch: integer, the number of steps (based on the batch size) that a full epoch will require.
    - epochs: integer, the number of epochs that the trainig process will run for.


"""

class Step_decay(Callback):

    """
    --------------------------- I N I T I A L I S A T I O N -------------------------
    """
    def __init__(self, min_lr=1e-6, max_lr=0.1, num_steps=5, steps_per_epoch=None, epochs=None):

      super().__init__()

      # [1] Initialise the class variables:
      # - min_lr
      # - max_lr
      # - total_iterations (steps_per_epoch * epochs)
      # - iteration (your counter)
      # - decrease which should be the (integer) value of the absolute difference between the max_lr - min_lr devided by the num_steps

      temp = int(self.total_iterations // num_steps)

      self.steps_pos = []

      for pos in range(temp, int(self.total_iterations), temp):
        self.steps_pos.append(pos)

    """
    --------------------------- E N D  O F  I N I T I A L I S A T I O N -------------------------
    """

    """
    --------------------------- F U N C T I O N  C A L C U L A T E _ N E W _ L R -------------------------
    """
    def calculate_new_lr(self):

      index = self.steps_pos.index(self.iteration) + 1

      # [2] return the max_lr minus the decrease times your iteration index
      return

    """
    --------------------------- E N D  F U N C T I O N -------------------------
    """


    """
    --------------------------- F U N C T I O N  O N _ T R A I N _ B E G I N -------------------------
    """
    def on_train_begin(self, logs=None):
      logs = logs or {}
      K.set_value(self.model.optimizer.lr, self.max_lr)

    """
    --------------------------- E N D  F U N C T I O N -------------------------
    """


    """
    --------------------------- F U N C T I O N  O N _ B A T C H _ E N D -------------------------
    """
    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}

        # [3] increase iteration counter by 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if (self.iteration in self.steps_pos):
            K.set_value(self.model.optimizer.lr, self.calculate_new_lr())

    """
    --------------------------- E N D  F U N C T I O N -------------------------
    """


"""
--------------------------------------------------------------------------------------------------------
                E N D   O F  C L A S S    S T E P _ D E C A Y
--------------------------------------------------------------------------------------------------------
"""
