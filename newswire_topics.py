"""
Trains and evaluates a simple multi-layer perceptron
on the Reuters news.
  (topic classification task)
"""

from __future__ import print_function
import numpy as np
np.random.seed(17)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


max_words = 7000

def load_vect_mat():
	"""The main method for loading, vectorizing, matrix-forming the newswire (labeled train & test) data
	to be fed into the Keras functional model API .fit and .evaluate functions.

	Arguments
	---------
	none

	Returns
	-------
	ttPair -- The usual pair of (X, Y)-train and that for test  (tuple/pair of tuples/pairs)
	"""

	print('\nLoading data...')
	(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
	print(len(X_train), 'train sequences  Be like:')
	print(X_train[0])
	print(len(X_test), 'test sequences  Be like:')
	print(X_test[0])

	global nb_classes
	nb_classes = np.max(y_train)+1
	print(nb_classes, 'topic classes')

	print('\nVectorizing (1/0) sequence data...')
	tokenizer = Tokenizer(nb_words=max_words)
	X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
	X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	print('\nConvert the list of (integer) class labels to one hotshot! -- 1/0 "row-wise" topic matrix (for use with categorical_crossentropy)')
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	print(y_train[0],' --> ',Y_train[0])
	print('... --> ...')
	print(y_train[-1],' --> ',Y_train[-1])
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	print('Y_train shape:', Y_train.shape)
	print('Y_test shape:', Y_test.shape)
	ttPair = ((X_train, Y_train), (X_test, Y_test))
	return ttPair

def main():
	(X_train, Y_train), (X_test, Y_test) = load_vect_mat()

	print('\nBuilding model...')
	model0 = Sequential([
		Dense(256, input_dim=max_words),
		Activation('softplus'), # the antiderivative of 'sigmoid'!
		Dropout(0.5),
		Dense(nb_classes),
		Activation('softmax')
	])
	model1 = Sequential()
	model1.add(	Dense(256, input_dim=max_words))#, init='uniform')	)
	model1.add(Activation('softplus'))
	model1.add(Dropout(0.5))
	model1.add(	Dense(256))#, init='uniform') )
	model1.add(Activation('tanh'))
	model1.add(Dropout(0.4))
	model1.add( Dense(nb_classes) )
	model1.add(Activation('softmax'))

	print("Summary of the model:")
	model1.summary()
	# the JSON representation of the model
	jsonStr = model1.to_json()
	print(jsonStr)
	print()
	import json
	model_json = json.loads(jsonStr)
	json.dump(model_json, open('reuters_mlp1.json', 'wb'), indent=3)

	from keras.optimizers import SGD
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# ended up turning off the 'sgd' solver..
	model1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

	nb_epoch = 7
	model1.fit(X_train, Y_train, nb_epoch=nb_epoch, validation_split=0.1)
	score = model1.evaluate(X_test, Y_test)
	print('\nTest score:', score[0])
	print('Test accuracy:', score[1])

if __name__ == '__main__':
	main()


