
from keras.datasets import mnist
from keras.layers import Input, Lambda
from base_network import BaseNetwork
from keras.models import Model
from utils import euclidean_distance, eucl_dist_output_shape
import json


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
input_dim = 784

# network definition
base_network = BaseNetwork(input_dim)

inp_a = Input(shape=(input_dim,))
inp_b = Input(shape=(input_dim,))

# we re-use the same instance `base_network`;
# the weights will be shared across the two branches!
out_a = base_network(inp_a)
out_b = base_network(inp_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([out_a, out_b])
super_model = Model(input=[inp_a, inp_b], output=distance)

jsonStr = super_model.to_json()
modelJson = json.loads(jsonStr)
json.dump(modelJson, open("siamese.json", 'wb'), indent=4)
