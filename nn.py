import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from tensorflow.examples.tutorials.mnist import input_data

#Loading the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Defining the placeholder
x = tf.placeholder(tf.float32, [None, 28,28,1]) ## A placeholder for the data

K=200
L=100
M=60
N=30

#First Layer
w1 = tf.Variable(tf.truncated_normal([28*28,K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

#Second Layer
w2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

#Thrid Layer
w3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

#Fourth Layer
w4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

#Fifth Layer
w5 = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

#reshaping the images
x = tf.reshape(x,[-1,28*28])

#Probability of keeping neurons for each layer (Defining the Placeholder)
pkeep = tf.placeholder(tf.float32)

#Defining the model for each layer
y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
y1 = tf.nn.dropout(y1,pkeep)

y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)
y2 = tf.nn.dropout(y2,pkeep)

y3 = tf.nn.relu(tf.matmul(y2,w3)+b3)
y3 = tf.nn.dropout(y3,pkeep)

y4 = tf.nn.relu(tf.matmul(y3,w4)+b4)
y4 = tf.nn.dropout(y4,pkeep)

y = tf.nn.softmax(tf.matmul(y4,w5)+b5)

#Defining the placeholder for the labels
Y_ = tf.placeholder(tf.float32, [None, 10])

#Defining the loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(y))

#Computing the accuracy 
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(Y_,1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Optimizing the loss function
optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy)

#Initiating the global variables
init = tf.global_variables_initializer()

#Running the Session
sess = tf.Session()
sess.run(init)

#Defining the accuracy matrices for training data and testing data 
acc_test_mat = []
acc_train_mat = []


for i in range(1000):

    #Feeding in training data
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {x:batch_X, Y_:batch_Y, pkeep:0.75}

    sess.run(train_step, feed_dict=train_data)
    #Appending training data accuracy for each epoch
    acc_train_mat.append(sess.run(acc, feed_dict = train_data) *100)

    #Feeding in testing data
    test_data = {x:mnist.test.images, Y_:mnist.test.labels, pkeep:0.75}

    #Appending testing data accuracy for each epich
    acc_test_mat.append(sess.run(acc, feed_dict = test_data)*100)


#Plotting accuracy for both training data (in orange) and testing data (in blue)
x_axis = np.linspace(0,1000,1000)
plt.plot(x_axis, acc_train_mat, 'orange')
plt.plot(x_axis, acc_test_mat, 'blue')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('Model Accuracy')
plt.legend(['training', 'testing'], loc='best')

plt.show()

