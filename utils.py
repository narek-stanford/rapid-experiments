
import numpy as np
from keras.utils import np_utils


def preprocess(X, y):
	# converting RGB -> BGR
	X = X[[2,1,0], :,:]

	y = np_utils.to_categorical(y, nb_classes=10)
	return (X, y)

X = np.random.rand(3, 7, 4)
y = np.zeros(len(X))
preprocess(X, y)
