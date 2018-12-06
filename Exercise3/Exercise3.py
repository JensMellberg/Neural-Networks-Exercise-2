
# coding: utf-8

# # Tensorflow Tutorial (MNIST with one hidden layer)
# ## Neural Networks (TU Graz 2018)
# (Adapted from the documentation of tensorflow, find more at: www.tensorflow.org)
#
#
# Improving the MNIST tutorial by adding one hidden layer
#
# <img src="hidden_layer.png" style="width: 200px;" />

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# Import dataset and libraries.
# Please ignore the deprecation warning while importing the MNIST dataset.

# In[11]:

from nn18_ex2_load import load_isolet
import tensorflow as tf
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import sys
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(X, C, X_tst, C_tst) = load_isolet()
#print(X.shape[0])
#print( X[1,:])
#print(np.mean(X[1,:]))
#print(np.std(X[1,:]))
#
res = False
if len(sys.argv) > 1 and sys.argv[1] == "res":
    res = True

for i in range(X.shape[0]):
    mean = np.mean(X[i,:])
    std = np.std(X[i,:])
    for f in range(X.shape[1]):
        X[i,f] = (X[i,f] - mean)/std

for i in range(X_tst.shape[0]):
    mean = np.mean(X_tst[i,:])
    std = np.std(X_tst[i,:])
    for f in range(X_tst.shape[1]):
        X_tst[i,f] = (X_tst[i,f] - mean)/std
#print(np.std(X[1,:]))
#print(np.std(X[5,:]))
#print(X[1,:])
C_matrix = np.zeros((6238,26))
for x in range(6238):
    C_matrix[x,C[x]-1] = 1

C_matrix_test = np.zeros((1559,26))
for x in range(1559):
    C_matrix_test[x,C_tst[x]-1] = 1

C = C_matrix
C_tst = C_matrix_test
# Define your variables and the operations that define the tensorflow model.
# - x,y,z do have have numerical values, those are symbolic **"Tensors"**
# - x is a matrix and not a vector, is has shape [None,784]. The first dimension correspond to a **batch size**. Multiplying larger matrices is usually faster that multiplying small ones many times, using minibatches allows to process many images in a single matrix multiplication.

# In[3]:



# Give the dimension of the data and chose the number of hidden layer
n_in = 300
n_out = 26
n_hidden = 40
learning_rate = 0.00001
layers = 9

# Set the variables

W_hid = []
b_hid = []
for i in range(layers):
    W_hid.append(tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True))
    b_hid.append(tf.Variable(np.zeros(n_hidden),trainable=True))

W_hid[0] = tf.Variable(rd.randn(n_in,n_hidden) / np.sqrt(n_in),trainable=True)

w_out = tf.Variable(rd.randn(n_hidden,n_out) / np.sqrt(n_in),trainable=True)
b_out = tf.Variable(np.zeros(n_out))

# Define the neuron operations
x = tf.placeholder(shape=(None,300),dtype=tf.float64)
z = 0
y =[]
y.append(tf.nn.relu(tf.matmul(x,W_hid[0]) + b_hid[0]))
if res:
    for i in range(1,layers/2):
        exit()
else:
    for i in range(1,layers):
        y.append(tf.nn.relu(tf.matmul(y[i-1],W_hid[i]) + b_hid[i]))
    z = tf.nn.softmax(tf.matmul(y[layers-1],w_out) + b_out)


# Define the loss as the cross entropy: $ - \sum y \log y'$

# In[4]:


z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))


# The operation to perform gradient descent.
# Note that train_step is still a **symbolic operation**, it needs to be executed to update the variables.
#

# In[23]:

#argument is learning rate
train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999, 0.00000001).minimize(cross_entropy)


# To evaluate the performance in a readable way, we also compute the classification accuracy.

# In[8]:


correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))


# Open a session and initialize the variables.

# In[7]:


init = tf.global_variables_initializer() # Create an op that will
sess = tf.Session()
sess.run(init) # Set the value of the variables to their initialization value


# In[24]:


# Re init variables to start from scratch
sess.run(init)

# Create some list to monitor how error decreases
test_loss_list = []
train_loss_list = []

test_acc_list = []
train_acc_list = []

# Create minibatches to train faster
k_batch = 20
X_batch_list = np.array_split(X,k_batch)
labels_batch_list = np.array_split(C,k_batch)
train_acc_final=0
test_acc_final=0
final_test_eval = 0
epochs=0
iterations = 500

for k in range(iterations):
    # Run gradient steps over each minibatch
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})

    train_loss = sess.run(cross_entropy, feed_dict={x:X, z_:C})
    train_acc = sess.run(accuracy, feed_dict={x:X, z_:C})

    test_loss = 0
    test_acc = 0
    test_loss = sess.run(cross_entropy, feed_dict={x:X_tst, z_:C_tst})
    test_acc = sess.run(accuracy, feed_dict={x:X_tst, z_:C_tst})

    #test_acc = sess.run(accuracy, feed_dict={x:X_tst, z_:C_tst})


    if test_acc > test_acc_final:
        train_acc_final=train_acc
        test_acc_final=test_acc
        epochs=k

    # Put it into the lists
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)

    if np.mod(k,10) == 0:
        print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))


# In[25]:



print("Training error")
print(train_acc_final)
print("Test error")
print(test_acc_final)
print("Epochs")
print(epochs)

fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

ax_list[0].set_xlabel('training iterations')
ax_list[1].set_xlabel('training iterations')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)
plt.show()
