from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


data = make_blobs(n_samples = 10000,n_features=2,centers=2,random_state=20)
features = data[0] 
labels = data[1]

print(features)

x_coor = features[:,0]
y_coor = features[:,1] 
y_labels = labels

print(x_coor)
print(y_coor)



m= tf.Variable(0.06)
b = tf.Variable(0.04)

batch_size = 8

x = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

z = tf.nn.sigmoid(m*x +b)


error = tf.reduce_sum(tf.square(yph - z))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    batches =10000
    
    for i in range(batches):
        rand_ind = np.random.randint(len(features), size =batch_size)
        
        feed = {x:x_coor[rand_ind],yph:y_labels[rand_ind]}
        
        sess.run(train, feed_dict = feed)
        
    model_m, model_b = sess.run([m,b])


print(model_m)
print(model_b)

plt.scatter(x_coor,y_coor, c=labels,cmap ='coolwarm')
plt.show()
x_input = 2
y_input = 6

y_input_coor = model_m*x_input +model_b

if(y_input >= y_input_coor):
    print("This is a red point")
else:
    print("This is a blue point")

plt.scatter(x_coor,y_coor, c=labels,cmap ='coolwarm')
z_model = model_m*x_coor +model_b
plt.plot(x_coor, z_model, color="black")
plt.show()

 

