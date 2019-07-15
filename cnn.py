import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from progressbar import ProgressBar
from tensorflow.examples.tutorials.mnist import input_data
import math

pbar = ProgressBar()

#Loading the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Defining the placeholder
x = tf.placeholder(tf.float32, [None, 28,28,1]) ## A placeholder for the data

K=4
L=8
M=12

w1 = tf.Variable(tf.truncated_normal([6,6,1,K],stddev=0.1))
b1 = tf.Variable(tf.ones([K])/10)

w2 = tf.Variable(tf.truncated_normal([5,5,K,L],stddev=0.1))
b2 = tf.Variable(tf.ones([L])/10)

w3 = tf.Variable(tf.truncated_normal([4,4,L,M],stddev=0.1))
b3 = tf.Variable(tf.ones([M])/10)

N=200

w4 = tf.Variable(tf.truncated_normal([7*7*M, N],stddev=0.1))
b4 = tf.Variable(tf.ones([N])/10)

w5 = tf.Variable(tf.truncated_normal([N, 10],stddev=0.1))
b5 = tf.Variable(tf.ones([10])/10)

pkeep = tf.placeholder(tf.float32)

y1 = tf.nn.relu(tf.nn.conv2d(x,w1, strides=[1,1,1,1], padding='SAME') +b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1,w2, strides=[1,2,2,1], padding='SAME') +b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2,w3, strides=[1,2,2,1], padding='SAME') +b3)

YY = tf.reshape(y3, shape = [-1, 7*7*M])

y4 = tf.nn.relu(tf.matmul(YY,w4)+b4)
y4 = tf.nn.dropout(y4, rate =1-pkeep)
Ylogits = tf.matmul(y4,w5)+b5
y = tf.nn.softmax(Ylogits)

#Defining the placeholder for the labels
Y_ = tf.placeholder(tf.float32, [None, 10])

#Defining the loss function
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_))

#Computing the accuracy 
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(Y_,1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Optimizer with learning rate decay
global_step = tf.placeholder(tf.int32)
learning_rate = 0.0001 + tf.train.exponential_decay(0.003, global_step, 2000, 1/math.e)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

#Initiating the global variables
init = tf.global_variables_initializer()

#Running the Session
sess = tf.Session()
sess.run(init)

#Defining the accuracy matrices for training data and testing data 
acc_test_mat = []
acc_train_mat = []

#Progress Bar for Vizualization of progress
print('Progress: ')


for i in pbar(range(10000)):

    #Feeding in training data
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {x:np.reshape(batch_X, [-1,28,28,1]), Y_:batch_Y, pkeep:0.75, global_step:i}

    sess.run(train_step, feed_dict=train_data)

    #Appending training data accuracy for each epoch
    acc_train_mat.append(sess.run(acc, feed_dict = train_data) *100)

    #Feeding in testing data
    test_data = {x:np.reshape(mnist.test.images, [-1,28,28,1]), Y_:mnist.test.labels, pkeep:1.00, global_step:i}

    #Appending testing data accuracy for each epoch 
    acc_test_mat.append(sess.run(acc, feed_dict = test_data)*100)


#Plotting accuracy for both training data (in orange) and testing data (in blue)
x_axis = np.linspace(0,10000,10000)
plt.plot(x_axis, acc_train_mat, 'blue')
plt.plot(x_axis, acc_test_mat, 'red')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Model Accuracy')
plt.legend(['training', 'testing'], loc='best')

plt.show()
 
'''
 100% training accuracy
 Just above 99% testing accuracy 
 See CNN_acc.png and CNN_acc1.png
'''