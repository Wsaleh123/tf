import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

print(x)
print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=100)

##clf.fit(x_train,y_train)
##y_pred = clf.predict(x_test)

##acc = metrics.accuracy_score(y_test, y_pred)

plt.scatter(x[:,0],x[:,1], color="black")
plt.scatter(x[:,0],x[:,2], color="red")
plt.scatter(x[:,0],x[:,3], color="green")
plt.scatter(x[:,0],x[:,4], color="gray")
plt.scatter(x[:,0],x[:,5], color="teal")
plt.scatter(x[:,0],x[:,6], color="brown")
plt.scatter(x[:,0],x[:,7], color="blue")

plt.show()

print(acc)