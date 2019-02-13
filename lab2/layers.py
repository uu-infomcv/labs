"""Code based on Jonne Engelberts implementation of ShakeShake [https://github.com/jonnedtc/Shake-Shake-Keras]"""

from keras import backend as K
from keras.layers import Layer


class ShakeShake(Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):

      # [1] Get the batch size by using the backend (K) and `.shape`

      alpha = K.random_uniform((batch_size, 1, 1, 1))

      beta = K.random_uniform((batch_size, 1, 1, 1))
      # shake-shake during training phase
      def x_shake():
        return beta * x + K.stop_gradient((alpha - beta) * x)

      # even-even during testing phase
      def x_even():
        return 0.5 * x

      # [2] return the appropriate function based on the phase (HINT: look at the .in_train_phase documentation in the keras.io/backend webpage)

    def compute_output_shape(self, input_shape):
        return input_shape
