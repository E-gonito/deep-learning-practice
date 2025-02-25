import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Layer 

# Get all devices, tell TF to only see GPUs (no CPU) 
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus, 'GPU')  

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
        return tf.matmul(input, self.weight) + self.bias
x = tf.ones((2,2))
layer1 = CustomDenseLayer(3)    
y = layer1(x)
print(y)
