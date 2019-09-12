from __future__ import absolute_import, division, print_function
import numpy as np 
from layer_utils import get_weights, get_biases, conv2d_layer,\
    pool2d_layer, flatten_layer, fully_connected_layer, \
        activation_layer
import tensorflow as tf

class VGG16Block:
    def __init__(self, input_shape, ouput_shape, weights, biases):
        self.input_shape = input_shape
        self.ouput_shape = ouput_shape
        self.image_input = tf.placeholder(
            dtype = tf.float32, shape = [None, self.input_shape, self.input_shape])
        self.feature_output = tf.placeholder(
            dtype = tf.int32, shape = [None, self.ouput_shape])
        self.weights = weights
        self.biases = biases

    

    