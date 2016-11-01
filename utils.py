
import numpy as np
from keras.utils import np_utils
from keras.models import load_model, Model
from keras.layers import Input, Dense
import numpy as np
import h5py


def preprocess(X, y):
	# converting RGB -> BGR
	X = X[[2,1,0], :,:]

	y = np_utils.to_categorical(y, nb_classes=10)
	return (X, y)

X = np.random.rand(3, 7, 4)
y = np.zeros(len(X))
preprocess(X, y)



def returnIdentical(model):
	model.save('my_model.h5')
	del model

	model = load_model('my_model.h5')
	# returns an identical (compiled!) model
	return model

x = Input(shape=(32,))
y = Dense(10)(x)
myModel = Model(input=x, output=y)
returnIdentical(myModel)



m1, m2, m3 = np.random.random(size=(1000,20)), np.random.random(size=(1000,200)), np.random.random(size=(1000,1000))
with h5py.File('data.h5', 'w') as hf:
    g1 = hf.create_group('grp1')
    g1.create_dataset('dset1', shape=None, dtype=None, data=m1, compression="gzip", compression_opts=9)
    g1.create_dataset('dset3', data=m3, compression="gzip", compression_opts=9)
 
    g2 = hf.create_group('grp2/sgrp')
    g2.create_dataset('dset2', data=m2, compression="gzip", compression_opts=9)