#!/usr/bin/env python3

# Classify delayed XOR in a {0,1} string, in input strings of VARIABLE length

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from grammar import *


def generate_data(nbr_of_examples):
    examples = np.ones((nbr_of_examples,50,7))
    targets = np.ones((nbr_of_examples,50,7))
    sl = np.ones(nbr_of_examples)
    for i in range(nbr_of_examples):
        current_string = make_embedded_reber();
        sl[i] = len(current_string)
        vec = str_to_vec(current_string)
        tar = str_to_next_embed(current_string)
        fullVec = np.zeros((50,7))
        fullTar = np.zeros((50,7))
        fullVec[:-(50-vec.shape[0]),:] = vec
        fullTar[:-(50-vec.shape[0]),:] = tar
        print(vec.shape)
        print(fullVec.shape)
        examples[i] = fullVec
        targets[i] = fullTar

    return examples, targets, sl


tf.reset_default_graph()  # for iPython convenience

# ----------------------------------------------------------------------
# parameters

sequence_length = 50
num_train, num_valid, num_test = 5000, 500, 500

#cell_type = 'simple'
#cell_type = 'gru'
cell_type = 'lstm'
num_hidden = 14

batch_size = 40
learning_rate = 0.01
max_epoch = 200

# ----------------------------------------------------------------------

# Generate delayed XOR samples
X_train, y_train, sl_train = generate_data(num_train)

X_valid, y_valid, sl_valid = generate_data(num_valid)

X_test, y_test, sl_test = generate_data(num_test)


# placeholder for the sequence length of the examples
seq_length = tf.placeholder(tf.int32, [None])

# input tensor shape: number of examples, input length, dimensionality of each input
# at every time step, one bit is shown to the network
X = tf.placeholder(tf.float32, [None, sequence_length, 7])

# output tensor shape: number of examples, dimensionality of each output
# Binary output at end of sequence
y = tf.placeholder(tf.float32, [None, 7])

# define recurrent layer
if cell_type == 'simple':
  cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
  # cell = tf.keras.layers.SimpleRNNCell(num_hidden) #alternative
elif cell_type == 'lstm':
  cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
elif cell_type == 'gru':
  cell = tf.nn.rnn_cell.GRUCell(num_hidden)
else:
  raise ValueError('bad cell type.')
# Cells are one fully connected recurrent layer with num_hidden neurons
# Activation function can be defined as second argument.
# Standard activation function is tanh for BasicRNN and GRU


# only use outputs, ignore states
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length) # NEW
# tf.nn.dynamic_rnn(cell, inputs, ...)
# Creates a recurrent neural network specified by RNNCell cell.
# Performs fully dynamic unrolling of inputs.
# Returns:
# outputs: The RNN output Tensor shaped: [batch_size, max_time, cell.output_size].

# get the unit outputs at the last time step
last_outputs = outputs[:,-1,:]

# add output neuron
y_dim = int(y.shape[1])
w = tf.Variable(tf.truncated_normal([num_hidden, y_dim]))
b = tf.Variable(tf.constant(.1, shape=[y_dim]))

y_pred = tf.nn.xw_plus_b(last_outputs, w, b)
# Matrix multiplication with bias

# define loss, minimizer and error
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

mistakes = tf.not_equal(y, tf.maximum(tf.sign(y_pred), 0))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# split data into batches

num_batches = int(X_train.shape[0] / batch_size)
X_train_batches = np.array_split(X_train, num_batches)
y_train_batches = np.array_split(y_train, num_batches)
sl_train_batches = np.array_split(sl_train, num_batches)

# train

error_train_ = []
error_valid_ = []

for n in range(max_epoch):
    print('training epoch {0:d}'.format(n+1))

    for X_train_cur, y_train_cur, sl_train_cur in zip(X_train_batches, y_train_batches, sl_train_batches):
        sess.run(train_step, feed_dict={X: X_train_cur, y: y_train_cur, seq_length: sl_train_cur})
        # We also need to feed the current sequence length
    error_train = sess.run(error, {X: X_train, y: y_train, seq_length: sl_train})
    error_valid = sess.run(error, {X: X_valid, y: y_valid, seq_length: sl_valid})

    print('  train:{0:.3g}, valid:{1:.3g}'.format(error_train, error_valid))

    error_train_ += [error_train]
    error_valid_ += [error_valid]

    if error_train == 0:
        break

error_test = sess.run(error, {X: X_test, y: y_test, seq_length: sl_test})
print('-'*70)
print('test error after epoch {0:d}: {1:.3f}'.format(n+1, error_test))

sess.close()

plt.figure()

plt.plot(np.arange(n+1), error_train_, label='training error')
plt.plot(np.arange(n+1), error_valid_, label='validation error')
plt.axhline(y=error_test, c='C2', linestyle='--', label='test error')
plt.xlabel('epoch')
plt.xlim(0, n)
plt.legend(loc='best')
plt.tight_layout()

plt.show()
