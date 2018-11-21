import tensorflow as tf
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

#Load audio import script
from nn18_ex2_load import load_isolet

X, C, X_tst, C_tst = load_isolet()

# C labels converted to a one-out-of-K coding
C_matrix = np.zeros((6238,26))
for x in range(6238):
    for y in range(26):
        if (y+1) == C[x]:
            C_matrix[x,y] = 1

C_matrix_tst = np.zeros((1559,26))
for x in range(1559):
    for y in range(26):
        if (y+1) == C_tst[x]:
            C_matrix_tst[x,y] = 1

# Create validation set
#X_val = X[0:1559,:]
#C_val = C_matrix[0:1559,:]

# Training set, not used for final testing
#X = X[1559:,:]
#C_matrix = C_matrix[1559:,:]


# Dimension of data and number of hidden layers
n_in = 300
n_hidden = 200
#n_hidden2 = 400
n_out = 26

# Weight and bias without hidden layers
#W = tf.Variable(rd.randn(300,26),trainable=True)
#b = tf.Variable(np.zeros(26),trainable=True)

# Set the variables Hidden 1
W_hid = tf.Variable(rd.randn(n_in,n_hidden) / np.sqrt(n_in),trainable=True)
print(W_hid.shape)
b_hid = tf.Variable(np.zeros(n_hidden),trainable=True)

# Set the variables Hidden 2
#W_hid2 = tf.Variable(rd.randn(n_hidden,n_hidden2) / np.sqrt(n_hidden),trainable=True)
#print(W_hid2.shape)
#b_hid2 = tf.Variable(np.zeros(n_hidden2),trainable=True)

w_out = tf.Variable(rd.randn(n_hidden,n_out) / np.sqrt(n_in),trainable=True)
b_out = tf.Variable(np.zeros(n_out))

# Define neuron operations
x = tf.placeholder(shape=(None,300),dtype=tf.float64)
y = tf.nn.tanh(tf.matmul(x,W_hid) + b_hid)
#y2 = tf.nn.tanh(tf.matmul(y,W_hid2) + b_hid2)
z = tf.nn.softmax(tf.matmul(y,w_out) + b_out)

z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
# Cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))

# Training step: Only updates variable when it is called
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

# Prediction and accuracy
correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# Create an op that will initialize the variable on demand
init = tf.global_variables_initializer()
sess = tf.Session()

# Init variables to start from scratch
sess.run(init)

# Create some list to monitor how error decreases
test_error_list = []
train_error_list = []
test_acc_list = []
train_acc_list = []

# Create mini batches to train faster
k_batch = 104
X_batch_list = np.array_split(X,k_batch)
labels_batch_list = np.array_split(C_matrix,k_batch)
train_acc_final=0
test_acc_final=0
epochs=0

# Number of Epochs
for k in range(59):
    # Run gradient steps over each minibatch
    for x_minibatch, labels_minibatch in zip (X_batch_list, labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, z_: labels_minibatch})



    # Compute error of the whole dataset
    train_err = sess.run(cross_entropy, feed_dict={x: X, z_: C_matrix})
    test_err = sess.run(cross_entropy, feed_dict={x: X_tst, z_: C_matrix_tst})

    # Compute the acc over the whole dataset
    train_acc = sess.run (accuracy, feed_dict={x: X, z_: C_matrix})
    test_acc = sess.run (accuracy, feed_dict={x: X_tst, z_: C_matrix_tst})
    if test_acc > test_acc_final:
        train_acc_final=train_acc
        test_acc_final=test_acc
        epochs=k
    # Put it into the lists
    test_error_list.append (test_err)
    train_error_list.append(train_err)
    test_acc_list.append (test_acc)
    train_acc_list.append (train_acc)
    #print(train_err)

    if np.mod (k, 10) == 0:
        print ('iteration {} test accuracy: {:.3f}'.format (k + 1, test_acc))



# Plot the error and accuracy

print("Training error")
print(train_acc_final)
print("Validation error")
print(test_acc_final)
print("Epochs")
print(epochs)

fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_error_list, color='blue', label='training', lw=2)
ax_list[0].plot(test_error_list, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)


ax_list[0].set_xlabel('training iterations')
ax_list[1].set_xlabel('training iterations')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)
plt.show()
