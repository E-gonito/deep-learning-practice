import tensorflow as tf           # Core library for building and running deep learning models
import numpy as np                # Library for numerical operations (arrays, random numbers)
from tensorflow.keras.layers import Layer, Dropout, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# Restrict TensorFlow's visible devices to GPU only (disables CPU as a compute device)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus, 'GPU')

# Custom layer subclassing the Keras Layer base class
class CustomDenseLayer(Layer):
    def __init__(self, num_outputs):
        super(CustomDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # Create trainable weight and bias based on input shape and desired outputs
        self.weight = self.add_weight(shape=(input_shape[-1], self.num_outputs),
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.num_outputs,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, input):
        # Defines the forward pass: output = input * weight + bias
        return tf.matmul(input, self.weight) + self.bias

# Create a Sequential model consisting of:
#  1) CustomDenseLayer with 128 neurons
#  2) Dropout layer with 0.5 dropout rate
#  3) Another CustomDenseLayer with 10 outputs (one per class)
#  4) Softmax activation to convert scores to probabilities
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10),
    Softmax()
])

# Generate random training data
x_train = np.random.random((1000, 20))   # 1000 samples, each with 20 features
y_train = np.random.randint(10, size=(1000, 1))  # Random labels in [0..9]

# Convert labels to one-hot vectors, e.g., label 3 -> [0,0,0,1,0,0,0,0,0,0]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy')

# Print model info before building it (no weights are set yet)
print("Model summary before building:")
model.summary()

# Force the model to build by providing an explicit input shape
model.build((1000, 20))

# Print the model summary after building (weights now allocated)
print("\nModel summary after building:")
model.summary()

# Plot the model architecture and save it to 'model_architecture.png'
plot_model(model,
           to_file='model_architecture.png',
           show_shapes=True,
           show_layer_names=True)
