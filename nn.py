## blue points are 0 and red points are 1

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


data = make_blobs(n_samples = 10000,n_features=2,centers=2,random_state=20)
features = data[0] 
labels = data[1]

x_coordinate = features[:,0]
y_coordinate = features[:,1]

plt.scatter(x_coordinate, y_coordinate, c=labels, cmap= 'coolwarm')
plt.show()
print(features)
print(labels)

## Neural Network

#Defining the variables
w = tf.Variable(np.random.rand(5000,2))
b = tf.Variable(.04)

#defining the batch_size
batch_size = 8

#Defining the placeholders
x = tf.placeholder(tf.float64, [batch_size]) #data to be fed
yph = tf.placeholder(tf.float64, [batch_size]) #labels


#Creating the model
y = tf.nn.sigmoid(w*x +b)

#defining the error function
error = tf.reduce_sum(tf.square(yph - y))

#defining the optimizer (essentially the task)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    batches =10000
    
    for i in range(batches):
        rand_ind = np.random.randint(len(features), size =batch_size)
        
        feed = {x:features[rand_ind],yph:labels[rand_ind]}
        
        sess.run(train, feed_dict = feed)

    model_w, model_b = sess.run([w,b])


 

