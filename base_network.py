
from keras.models import Sequential
from keras.layers import Dropout, Dense


def BaseNetwork(input_dim):
    seq = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu'),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(128, activation='relu')
    ])
    return seq


def main():
	bn = BaseNetwork(74)
	bn.summary()
if __name__ == '__main__':
	main()