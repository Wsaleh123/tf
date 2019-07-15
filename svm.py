#red=1, blue=0
from sklearn.datasets import make_blobs
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = make_blobs(n_samples = 10000,n_features=2 ,centers=2)
features = data[0] 
labels = data[1]

x_coor = features[:,0]
y_coor = features[:,1]

coor =[]

for i in range(10000):
    coor.append([x_coor[i],y_coor[i]])

X_train, X_test, Y_train, Y_test = skl.model_selection.train_test_split(coor,labels, test_size=0.3)


clf = svm.SVC(kernel='poly',degree=3)
knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')

clf.fit(X_train, Y_train)
knn.fit(X_train, Y_train)


svm_predictions = clf.predict(X_test)
knn_predictions = knn.predict(X_test)

print('SVM Accuracy: ' + str(metrics.accuracy_score(svm_predictions,Y_test)*100) + '%')
print('KNN Accuracy: ' + str(metrics.accuracy_score(knn_predictions,Y_test)*100) + '%')


plt.scatter(x_coor,y_coor, c=labels, cmap = 'coolwarm')

plt.show()
