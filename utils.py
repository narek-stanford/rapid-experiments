
from __future__ import print_function
import numpy as np
from keras.utils import np_utils
from keras.models import load_model, Model
from keras.layers import Input, Dense
import numpy as np
import h5py
from os import path
import csv


def csv_contents2list_of_tuples(CSV_filename):
	with open(CSV_filename) as of:
		# stands for 'list of tuples'..
		lots = list(tuple(line) for line in csv.reader(of, delimiter=','))
	return lots
LOT = csv_contents2list_of_tuples(path.expanduser('~')+"/Desktop/VI/skechers_all_shoes_train.csv")
print('Total number of tuples:',len(LOT))
tupSize = len(LOT[0])
print('Each tuple is a '+str(tupSize)+'-tuplet')

chunkSize = 7
nb_sample = chunkSize*14
screens = LOT[:nb_sample]
print(screens)


def generate_chunks(sample_tuples):
	for chunk_index in range(len(sample_tuples)/chunkSize):
		st,upto = chunk_index*chunkSize, (chunk_index+1)*chunkSize
		curChunk = screens[st:upto]
		yield curChunk


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





