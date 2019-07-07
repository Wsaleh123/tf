from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

data = make_blobs(n_samples=2500, n_features=3, centers=3, random_state=20)

features = data[0]
labels = data[1]

print(np.ndarray.shape(data=data))


plt.scatter(features[:,0], features[:,1], c=labels, cmap='coolwarm')
plt.scatter(features[:,0], features[:,2], c=labels, cmap='coolwarm')


plt.show()


