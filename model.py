import tensorflow as tf     # Core TensorFlow library
import numpy as np    
from tensorflow import keras            # NumPy for numerical operations
from keras.utils import plot_model     # For visualizing model architecture
from keras.layers import Layer         # Base class for creating custom layers
from keras.models import Sequential    # Sequential model API
from keras.layers import Softmax       # Softmax activation layer
from keras.layers import Dropout  

class CustomDenseLayer(Layer):
    def __init__(self, num_outputs):
        super(CustomDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.num_outputs),
                                initializer='random_normal',
                                trainable=True)
        self.bias = self.add_weight(shape=(self.num_outputs,),
                                initializer='zeros',
                                trainable=True)

    def call(self, input):
        return tf.matmul(input, self.kernel)