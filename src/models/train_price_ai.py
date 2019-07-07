import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from pandas import DataFrame
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.engine import training
from typing import Dict, Tuple
from tensorflow.python.keras.callbacks import TensorBoard


class PriceAiTrainer:

    def __init__(
            self,
            data_by_symbol: Dict[str, DataFrame],
            look_back=40,
            forward_prediction=40,
            test_size_percent=0.25,
            validation_column='close',
            log_dir='../logs/{}'
    ):
        self.tf_board = TensorBoard(log_dir=log_dir.format(datetime.now().isoformat()))
        self.sample_size = look_back * 2
        self.validation_column = validation_column
        self.test_size_percent = test_size_percent
        self.look_back = look_back
        self.forward_prediction = forward_prediction
        self.data_by_symbol = data_by_symbol
        self.num_features = 0
        self.validate_data_by_symbol = defaultdict()
        self.train_data_by_symbol = defaultdict()

        if len(data_by_symbol.keys()):
            self.num_features = len(data_by_symbol[list(data_by_symbol.keys())[0]].columns) - 1
            self.__split_train_and_test_data()

    def build_model(self, neurons_first_layer=None, neurons_second_layer=None):
        neurons_first_layer = neurons_first_layer or self.look_back * self.num_features
        neurons_second_layer = neurons_second_layer or int(neurons_first_layer / 2)

        model = Sequential()
        model.add(
            LSTM(neurons_first_layer, input_shape=(self.look_back, self.num_features), return_sequences=True)
        )
        model.add(LSTM(neurons_second_layer, input_shape=(neurons_first_layer, 1)))
        model.add(Dense(self.forward_prediction))
        return model

    def get_training_data(self, symbol):
        return self.train_data_by_symbol[symbol]

    # Build the model
    def train(self, model: training.Model, symbol, epochs=220):
        x_train, y_train = self.train_data_by_symbol[symbol]
        x_validate, y_validate = self.validate_data_by_symbol[symbol]
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        history = model.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_validate, y_validate), shuffle=True,
            verbose=2, callbacks=[self.tf_board]
        )
        model.save('../models/price_ai__{}'.format(datetime.now().isoformat()))
        return model, history

    def reshape_data(self, data: DataFrame):

        num_items = len(data)
        # make the dataset evenly sized
        items_to_drop = int((num_items % self.look_back))
        train_data = data.copy()[items_to_drop:][:-self.look_back]
        validation_data = data[[self.validation_column]].copy().iloc[(items_to_drop + self.look_back):]

        print(items_to_drop)

        x_data = np.array(np.split(train_data.to_numpy()[:, 1:], len(train_data) / 40))
        y_data = np.array(np.split(validation_data.to_numpy().flatten(), len(validation_data) / 40))

        return x_data, y_data

    def __split_train_and_test_data(self):
        for symbol, data in self.data_by_symbol.items():
            test_size = int(self.test_size_percent * len(data))
            self.train_data_by_symbol[symbol] = self.reshape_data(data[:-test_size])
            self.validate_data_by_symbol[symbol] = self.reshape_data(data.tail(test_size))
