from sklearn.datasets import make_blobs
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


data = make_blobs(n_samples = 10000,n_features=2 ,centers=2)
features = data[0] 
labels = data[1]

x_coor = features[:,0]
y_coor = features[:,1]

coor =[]

for i in range(10000):
    coor.append([x_coor[i],y_coor[i]])

coor = skl.model_selection.train_test_split(coor, test_size=0.3)
labels1 = skl.model_selection.train_test_split(labels, test_size = 0.3)

train_data = coor[0]
test_data = coor[1]

train_lbls = labels1[0]
test_lbls= labels1[1]


clf = svm.SVC(kernel='poly', gamma = .0001, C=1000, degree=3)
clf.fit(train_data, train_lbls)

acc = 0
correct = 0
total = 0

for i in range(3000):
    if (clf.predict([test_data[i]]) == test_lbls[i]):
        correct += 1
    total+=1

acc = (correct/total)*100
print("Accuracy: " +str(acc) + "%")

plt.scatter(x_coor,y_coor, c=labels, cmap = 'coolwarm')
plt.show()
