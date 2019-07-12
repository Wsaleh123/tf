#human = 1 horse=0
from os import listdir
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

#Converting from rgb to gray scale 
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#Initializing data and labels matrix
data = list()
labels = list()

#loading the horse data into the matrices
for filename in listdir('horse-or-human (1)/horses'):

    img_data = image.imread('horse-or-human (1)/horses/' + filename)
    data.append(rgb2gray(img_data))
    labels.append([1,0]) #Appending the labels to labels matrix
	
#loading images for humans
for filename in listdir('horse-or-human (1)/humans/'):
    img_data = image.imread('horse-or-human (1)/humans/' + filename)
    data.append(rgb2gray(img_data))
    labels.append([0,1])

#Combining data and labels matrices
combined = list(zip(data, labels))
#Shuffling the data
random.shuffle(combined)
#Distributing the data and labels
data, labels = zip(*combined)

#Grabbing 100 samples for training and testing 
data = data[:100][:][:]
labels = labels[:100][:]

#Performing a train test split on the data with 30% test size
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.3)



x = tf.placeholder(tf.float32, [None, 300,300]) ## A placeholder for the data
K=200
L=100
M=60
N=30

#First Layer
w1 = tf.Variable(tf.truncated_normal([300*300,K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

#Second Layer 
w2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

#Third Layer
w3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

#Fourth Layer
w4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

#Fifth Layer
w5 = tf.Variable(tf.truncated_normal([N,2],stddev=0.1))
b5 = tf.Variable(tf.zeros([2]))

#Reshaping the images 
X = tf.reshape(x,[-1,300*300])

#Defining the model
y1 = tf.nn.relu(tf.matmul(X,w1)+b1)
y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)
y3 = tf.nn.relu(tf.matmul(y2,w3)+b3)
y4 = tf.nn.relu(tf.matmul(y3,w4)+b4)
y = tf.nn.softmax(tf.matmul(y4,w5)+b5) #Final model

#Defining the placeholder for the labels
Y_ = tf.placeholder(tf.float32, [None, 2])

#Defining the loss function
cross_entropy = -tf.reduce_sum(Y_*tf.log(y))

#Computing the accuracy 
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(Y_,1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Defining the optimizer to optimize loss function
optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy)

#Initializing the global variables
init = tf.global_variables_initializer()

#Running the session
sess = tf.Session()
sess.run(init)

#List for accuracy of each epoch
acc_mat = []

#Training the model
for i in range(1000):
    train_data = {x:X_train, Y_:Y_train}
    sess.run(train_step, feed_dict=train_data)
    acc_mat.append(sess.run(acc, feed_dict = train_data))

#Testing the model with one sample from test data
test_data = {x:X_test, Y_:Y_test}
a = sess.run(y, feed_dict=test_data)

#Vizualization for the test image prediction
plt.imshow(X_test[0][:][:])
prediction = np.argmax(a[0])
if(prediction == 1):
    print('Prediction: Human')
else:
    print('Prediction: Horse')
print('Accuracy: ' + str(a[0][prediction]))


#Plotting accuracy graph for training data
x_axis = np.linspace(0,1000,1000)
plt.plot(x_axis, acc_mat, 'orange')

plt.show()
