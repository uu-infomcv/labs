# Convolutional Neural Networks - Practical session

### About

* In this practical we will be looking at how Convolutions are performed in Keras and how to also use custom learning rates.

The main tasks are listed as followed:

- [ ] How to use Keras's `layers.Conv2D` function to create a CNN.

- [ ] Experiment with custom layers and normalisation.

- [ ] Learn to create `Callback` functions that can be used during training to change the internal state of the model.

- [ ] Create a learning rate step decay that after a fixed number of iterations decreases the learning rate by a certain value.

This practical does assume that you have successfully completed the *Introduction to Neural Computations* practical session. So you will be required to copy your answers for `paths_list_from_directory`, `load_image` and `DataGenerator` functions over to the python script. Also keep in mind that we will be using 3 channel images so you may be required to adjust your code in the `load_image` function.

### Building CNNs with Keras

As we saw in the previous session, the starting point for creating a model is defining the shape of the input. In this case instead of using a vector as our data, the input will be batches of images. This means that the `Input` layer should be initialised with the same dimensions as the images. This can be done by setting the `shape` parameter of the function to 48x48x3. Start by opening `cnn.py`.

### Convolutional layers

Performing spatial Convolutions over images can be done with:

```
x = Conv2D(filters, kernel_size, strides=(1,1), padding='valid')(inputs)
```

This creates a convolution kernel that convolves the inputs to produce a tensor of outputs. The following function arguments should be taken to consideration when calling the function:

* **filters**: Should be an integer that is the number of kernels to be used. This will also correspond to the depth of the activation maps produced.
* **kernel_size**: Could be an integer or a tuple and defines the dimensions of the convolutional kernel. If you want to use kernels of unequal dimensions `(i.e. kernel_width != kernel_height)` then the shape should be defined as a tuple.
* **strides**: Similarly, the stride number can be an int/tuple if you would like to change the rate with which regions are convolved.
* **padding**: This specifies the zero padding to be done in order to keep or reduce the dimensionality of the output (compared to the input). It can take either of two values: `"valid"` if no padding is to be added or `"same"` to preserve the dimensions.

It should be noted that there is an option to adjust the way convolutions are performed based on the input shape. For example, if the colour channels are the first dimension in the input (e.g. 3x48x48) then you can adjust the **data_format** argument in the function with the `"channels_first"` value. - keep in mind that TensorFlow uses `"channels_last"` by default.

The output of the function is a 4D tensor of shape `(batch_size, width, height, filters)`

![](https://cdn-images-1.medium.com/max/1600/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)


In the code snipped we chose to use three Convolutions of with 32, 64 and 128 filters respectively.
```
# First block
x = Conv2D(filters=32, kernel_size=(3, 3), name='cnn_lab_conv1')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

# Second block
x = Conv2D(filters=64, kernel_size=(3, 3), name='cnn_lab_conv2')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

# Third block
x = Conv2D(filters=128, kernel_size=(3, 3), name='cnn_lab_conv3')(x)
x = BatchNormalization(axis=bn_axis)(x)
x = Activation('relu')(x)

x = Flatten()(x)

predictions = Dense(classes, activation='softmax', name='predictions')(x)
```
Notice that in Keras all the layers can also use the argument (if provided) `name`. This is especially useful in cases where you need to load wights from only specific layers so you can use the `by_name` argument of the `load_wights()` function.

### Pooling layers

There are build in functions for performing Max and Average Pooling in Keras:

```
x = MaxPooling2D(pool_size=(3,3), strides=None, padding='valid')(x)
```

Similarly to Convolutions, the input shape for pooling layers should be 4D tensors which are also the same shape for the output. The `pool_size` argument could be an integer or tuple.

Add two pooling layers after the first and second block in the script with pooling sizes of (3,3). You should see:

A) A decrease in the number of parameters for your network (roughly only 95K with pooling).

B) That the network trains much faster now.

C) That you should now have better test results.

Experiment by adding more Convolutions and occasionally adding Pooling operations keeping the number of parameters sensible and below those of initial network (545730). You can also add 1x1 Convolutions in order to increase your network depth.

### Global Pooling vs Flattening

A special case of pooling is Global Pooling in which all the dimensions are averaged/maxed to a single feature vector. This means that it takes a 4D tensor of type `(batch_size, width, heigh, filters)` as an input and uses pooling operations to create a 2D tensor of shape `(batch_size, depth/channels)`

```
x = GlobalAvgPool2D()(x)
```

Another way of vectorising inputs is to use a `Flatten` layer. In contrast to `GlobalPool`, this operation simply does a transformation of the data meaning that the output is simply all the features stacked to a single vector. For example:

```
inputs = Input(shape=(48,48,3))

x = Conv2D(64, (3,3), padding='valid')(inputs)
# This will produce an output shape of batchx46x46x64

x = Flatten()(x)
# This will create a vector of batchx135424 (46x46x64 = 135424)
```

It is recommended, in order to optimise the usage of your network parameters to have a CNN architecture that at the very last layers uses very (spatially) small activation maps (e.g. 4x4xfilters) and uses a `GlobalPooling` method for converting tensors to vectors.

This means that the same amount of extracted features are represented more robustly as Flatten can create a large number of parameters and computational overhead (think about this at your own time).

Replace the `Flatten()` layer in the network with a `GlobalAvgPool2D` layer and record the observed differences.

### Custom layers

As of the new version of Keras (2.0) when creating a custom layer, only three methods need to be implemented:
* `build(input_shape)`: Which is used to define any layer weights and at the end of the definition, it should set the `build` parameter of the layer to `True`. This can be done by calling `super([Layer], self).build()`.
* `call(x)`: The main logic and the overall functions of the layer should be written within this function. The passed argument should be the input tensor.
* `compute_output_shape(input_shape)`: This is used in the cases where the layer does indeed modify the dimensions of the input (e.g. as done in pooling or convolutions). The way the shape is transformed should be declared here.

Open the `layers.py` file. Based on the ShakeShake paper [[arxiv link]](https://arxiv.org/pdf/1705.07485.pdf) we will attempt to create a much simpler version of the regularisation that does not include residual connections.

Have a look at the class named `ShakeShake` and try to understand what each function does based on the requirements described.

Take note of the operation performed in the `call` function:

```
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
```  

Once completing the required lines of code add an instance of the ShakeShake layer in your model (you can also experiment with different locations).

### Creating custom callbacks

Open the file `lr_callbacks.py`. As you will see, you can create a custom callback function by extending the base class `callbacks.Callback`. The base class has access to the model's properties through `self.model`. The fist task will be to log the internal state of the model at the end of each epoch.

Start with the `InternalStateHistory` class:
```
class InternalStateHistory(Callback):

    def on_train_begin(self, logs={}):

      # [1] initialise the logged losses to an empty list

      # [2] initialise the logged learning rates to an empty list

      # [4] initialise the logged accuraccy rates to an empty list

      # [5] initialise the current iterration number to zero


    def on_epoch_end(self, batch, logs={}):

      # [1] append to the losses array the current loss by calling logs.get('loss')

      # [2] append to the lr values array the current lr by calling from keras backend (imported as K) the `get_value` function with `self.model.optimizer.lr` as argument.

      # [3] append the accuracy rate to the currosponding array similarly to how you obtained the loss.

      # [4] get the iteration number from self.model.optimizer - as you did for the lr

```
The purpose of this class is to keep track of the model loss, accuracy, the learning rate (if a non-constant function is used for the updates) and the iteration number (based on which these values are logged on).

The last task in this session will be to create a custom learning rate that decreases over time during training with a fixed number of steps. To do this, a class `Step_decay` has been (partially) implemented.

```
class Step_decay(Callback):
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

    def calculate_new_lr(self):

      index = self.steps_pos.index(self.iteration) + 1

      # [2] return the max_lr minus the decrease times your iteration index
      return

    def on_train_begin(self, logs=None):
      logs = logs or {}
      K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}

        # [3] increase iteration counter by 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if (self.iteration in self.steps_pos):
            K.set_value(self.model.optimizer.lr, self.calculate_new_lr())

```
Once finished, use the Step_decay with your model.

(Optional) Try and create a more mathematically complex learning rate reduction function (e.g. exponential reduction).
