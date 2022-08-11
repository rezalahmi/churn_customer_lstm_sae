import tensorflow as tf
from tensorflow import keras


class lstm_sae:

    def __init__(self, node_layer1, node_layer2, node_layer3):
        self.epoch = 50
        timesteps = 80000  # number of sample
        n_features = 1024  # number of features
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.LSTM(node_layer1, activation='relu', input_shape=(timesteps, n_features),
                                         return_sequences=True))
        self.model.add(keras.layers.LSTM(node_layer2, activation='relu', return_sequences=False))
        self.model.add(keras.layers.LSTM(node_layer3, activation='relu', return_sequences=False))
        self.model.add(keras.layers.RepeatVector(timesteps))
        self.model.add(keras.layers.LSTM(node_layer3, activation='relu', return_sequences=True))
        self.model.add(keras.layers.LSTM(node_layer2, activation='relu', return_sequences=True))
        self.model.add(keras.layers.LSTM(node_layer1, activation='relu', return_sequences=True))
        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(n_features)))

    def compile(self):
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X):
        self.model.fit(X, X, epochs=self.epoch, verbose=0)

    def predict(self, seq_in):
        return self.model.predict(seq_in, verbose=0)
