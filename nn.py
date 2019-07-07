import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28,28,1]) ## A placeholder for the data

K=200
L=100
M=60
N=30

w1 = tf.Variable(tf.truncated_normal([28*28,K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

w2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

w3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

w4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

w5 = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

x = tf.reshape(x,[-1,28*28])

y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)
y3 = tf.nn.relu(tf.matmul(y2,w3)+b3)
y4 = tf.nn.relu(tf.matmul(y3,w4)+b4)
y = tf.nn.softmax(tf.matmul(y4,w5)+b5)

Y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(Y_*tf.log(y))
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(Y_,1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))
summary = tf.summary.scalar("Loss", acc)
optimizer = tf.train.AdamOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

accuracy_mat = []
cost_mat = []
writer = tf.summary.FileWriter("logs", sess.graph)
for i in range(1000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {x:batch_X, Y_:batch_Y}

    sess.run(train_step, feed_dict=train_data)

    a,c, b = sess.run([acc, cross_entropy, summary], feed_dict = train_data)
    writer.add_summary(b, i)
    accuracy_mat.append(a*100)
    cost_mat.append(c)

batchX, batch_Y = mnist.test.next_batch(1)
train_data = {x:batch_X, Y_:batch_Y}
sess.run(train_step, feed_dict=train_data)

v, g = sess.run([x,y], feed_dict=train_data)
v = v[0].reshape(28,28)
v = np.stack((v,)*3, axis=-1)
plt.imshow(v)
prediction = np.argmax(g[0])
accuracy = g[0][prediction] *100
print("Prediction: ", prediction)
print("Accuracy: ", accuracy)


plt.show()