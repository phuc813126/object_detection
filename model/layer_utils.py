from __future__ import absolute_import, division, print_function
import tensorflow as tf

def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

def get_biases(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))

def conv2d_layer(layer, kernel_size, input_shape, output_shape, strides):
    weights = get_weights([kernel_size, kernel_size, input_shape, output_shape])
    biases = get_biases([output_shape])
    stride_shape = [1, strides, strides, 1]
    layer = tf.nn.conv2d(layer, weights, strides=stride_shape, padding="SAME") + biases
    return layer

def pool2d_layer(layer, kernel_size, strides):
    kernel = [1, kernel_size, kernel_size, 1]
    stride_shape = [1, strides, strides, 1]
    layer = tf.nn.max_pool(layer, ksize=kernel, strides=stride_shape, padding='SAME')
    return layer

def flatten_layer(layer):
    #convert to 1 dimensional
    input_size = layer.get_shape().as_list()
    new_size = input_size[-1] * input_size[-2] * input_size[-3]
    return tf.reshape(layer, [-1, new_size]), new_size

def fully_connected_layer(layer, input_shape, output_shape):
    weights = get_weights([input_shape, output_shape])
    biases = get_biases([output_shape])
    layer = tf.matmul(layer,weights) + biases  
    return layer

def activation_layer(layer):
    return tf.nn.relu(layer)