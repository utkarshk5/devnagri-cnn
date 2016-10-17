from skimage import io
import numpy as np
import tensorflow as tf
import sys, os, scipy

def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

train_path = 'train/'
valid_path = 'valid/'
px = 80
n_train, n_valid = 17205, 1829
n_iters = 10000
n_labels = 104

#####################################
# initializing tensorflow variables #
#####################################

x = tf.placeholder(tf.float32, [None, px**2])
y_ = tf.placeholder(tf.int64, [None])

W1 = tf.Variable(tf.truncated_normal([px**2, px**2], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[px**2]))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([px**2, px**2], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[px**2]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([px**2, n_labels], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[n_labels]))

y = tf.matmul(h2, W3) + b3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

############################################
# variables initialized, time to read files#
############################################

train_labels = []
valid_labels = []

with open(train_path + 'labels.txt') as f:
	train_labels = f.readlines()

with open(valid_path + 'labels.txt') as f:
	valid_labels = f.readlines()

train_labels = np.array([int(i) for i in train_labels])
train_index = [i for i in range(len(train_labels)) if train_labels[i] < n_labels]
train_labels = np.array([train_labels[i] for i in train_index])

valid_labels = np.array([int(i) for i in valid_labels])
valid_index = [i for i in range(len(valid_labels)) if valid_labels[i] < n_labels]
valid_labels = np.array([valid_labels[i] for i in valid_index])

train = np.array([scipy.misc.imresize(io.imread(train_path+str(i)+'.png'), (px/320.0)) for i in train_index], dtype='float32')
train = np.array([[(255.0-item)/255.0 for sublist in l for item in sublist] for l in train])

valid = np.array([scipy.misc.imresize(io.imread(valid_path+str(i)+'.png'), (px/320.0)) for i in valid_index], dtype='float32')
valid = np.array([[(255.0-item)/255.0 for sublist in l for item in sublist] for l in valid])

#################################
# files read, time to train now #
#################################

for i in range(n_iters):
	batch_xs = train[100*(i%(n_labels+1)) : 100*(i%(n_labels+1)+1)]
	batch_ys = train_labels[100*(i%(n_labels+1)) : 100*(i%(n_labels+1)+1)]
	if i%(n_labels+1) == 0:
		shuffle_in_unison(train, train_labels)
	if i%100 == 0:
		train_accuracy = sess.run(accuracy, feed_dict={x: train, y_: train_labels})
		valid_accuracy = sess.run(accuracy, feed_dict={x: valid, y_: valid_labels})
		print("step %d, training accuracy %g validation accuracy %g"%(i, train_accuracy, valid_accuracy))	
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

