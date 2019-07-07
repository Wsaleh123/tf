#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Sunday July 7 00:18:34 2019

@author: WesamSaleh
'''

# importing all necessesary packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data

# Loading the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28,28,1]) ## A placeholder for the data

#Shape parameters
K=200
L=100
M=60
N=30

#1st layer weights and biases
w1 = tf.Variable(tf.truncated_normal([28*28,K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

#2nd layer weights and biases
w2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

#3rd layer weights and biases
w3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

#4th layer weights and biases
w4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

#5th layer weights and biases
w5 = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

#reshaping the input data
x = tf.reshape(x,[-1,28*28])

#Creating the model for all layers -- last layer switches to softmax
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)
y3 = tf.nn.relu(tf.matmul(y2,w3)+b3)
y4 = tf.nn.relu(tf.matmul(y3,w4)+b4)
y = tf.nn.softmax(tf.matmul(y4,w5)+b5)

#Defining the placeholder for the labels
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(y))

#seeing if the weights matrix index with the highest value is the correct number
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(Y_,1))

#computing the accuracy
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#For tensorboard -- plotting the accuracy 
summary = tf.summary.scalar("Loss", acc)

#Difining the optimizer and minimizing the loss function
optimizer = tf.train.AdamOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

#initiating variables
init = tf.global_variables_initializer()

#Running the session
sess = tf.Session()
sess.run(init)

#Writing the accuracy data to logs file for tensorboard
writer = tf.summary.FileWriter("logs", sess.graph)

#Looping for 1000 epochs
for i in range(1000):
    batch_X, batch_Y = mnist.train.next_batch(100) #getting the training data and labels from mnist dataset (batch_size =100)
    train_data = {x:batch_X, Y_:batch_Y} #Dictionary for data with corresponding labels

    sess.run(train_step, feed_dict=train_data) #running gradient descent optimizer with a feed_dict of trian_data

    a,c, b = sess.run([acc, cross_entropy, summary], feed_dict = train_data) #running the accuracy, cost, and summary (for tensorboard)
    writer.add_summary(b, i) #adding the summary to tensorboard
    

batchX, batch_Y = mnist.test.next_batch(1) #Classification on 1 image for testing 
train_data = {x:batch_X, Y_:batch_Y} #image and label for testing
sess.run(train_step, feed_dict=train_data) #running the session for train data

v, g = sess.run([x,y], feed_dict=train_data) 
v = v[0].reshape(28,28)
v = np.stack((v,)*3, axis=-1)
plt.imshow(v)
prediction = np.argmax(g[0])
accuracy = g[0][prediction] *100
print("Prediction: ", prediction)
print("Accuracy: ", accuracy)


plt.show()