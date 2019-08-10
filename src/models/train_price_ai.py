import sys
from collections import defaultdict
from functools import reduce

from datetime import datetime

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.python.keras.engine import training
from typing import Dict, Tuple
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint


class BasePriceAi:

    def __init__(self, validation_column, look_back=5, forward_prediction=5):
        self.look_back = look_back
        self.validation_column = validation_column
        self.forward_prediction = forward_prediction

    def reshape_data(self, data: DataFrame):
        num_items = len(data)
        # make the dataset evenly sized
        items_to_drop = int((num_items % self.look_back))
        train_data = data.copy().iloc[items_to_drop:]
        validation_data = data[[self.validation_column]].copy().iloc[items_to_drop:]

        train_data = train_data.iloc[:-self.look_back]
        validation_data = validation_data.iloc[self.look_back:]

        scaler = MinMaxScaler(copy=False, feature_range=(0, 1))

        scaled_train_data = scaler.fit_transform(train_data.to_numpy()[:, 1:])  # drop index column
        scaled_validation_data = scaler.fit_transform(validation_data.to_numpy())

        x_data = np.array(np.split(scaled_train_data, len(train_data) / self.forward_prediction))
        y_data = np.array(np.split(scaled_validation_data, len(validation_data) / self.forward_prediction))

        # self.validate_x_y_integrity(x_data, y_data)
        return x_data, y_data

    def validate_x_y_integrity(self, x_data: np.numarray, y_data: np.numarray):
        """ Make sure the closing price """
        if x_data[1][0][0] != y_data[0][0][0]:
            raise Exception('Test data and validation is not aligned')


class PriceAiTrainer(BasePriceAi):

    def __init__(
            self,
            data_by_symbol: Dict[str, DataFrame],
            look_back=40,
            forward_prediction=40,
            test_size_percent=0.5,
            validation_column='close',
            log_dir='../logs/{}'
    ):
        super().__init__(look_back, forward_prediction=forward_prediction, look_back=look_back)
        self.tf_board = TensorBoard(log_dir=log_dir.format(datetime.now().isoformat()))
        self.sample_size = look_back * 2
        self.validation_column = validation_column
        self.test_size_percent = test_size_percent
        self.look_back = look_back
        self.data_by_symbol = data_by_symbol
        self.num_features = 0
        self.validate_data_by_symbol = defaultdict()
        self.train_data_by_symbol = defaultdict()

        if len(data_by_symbol.keys()):
            self.num_features = len(data_by_symbol[list(data_by_symbol.keys())[0]].columns) - 1
            self.__split_train_and_test_data()

    def build_model(self, neurons_first_layer=None):
        model = keras.Sequential()
        model.add(LSTM(64, input_shape=(self.look_back, self.num_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dense(1))
        return model

    def get_training_data(self, symbol):
        return self.train_data_by_symbol[symbol]

    # Build the model
    def train(self, model: training.Model, epochs=100):
        x_train, y_train = reduce(
            lambda all, values: (
                np.concatenate((all[0], values[0]), axis=0), np.concatenate((all[1], values[1]), axis=0)),
            self.train_data_by_symbol.values()
        )
        x_validate, y_validate = reduce(
            lambda all, values: (
                np.concatenate((all[0], values[0]), axis=0), np.concatenate((all[1], values[1]), axis=0)),
            self.validate_data_by_symbol.values()
        )
        x_validate, y_validate = self.shuffle(x_validate, y_validate)
        x_train, y_train = self.shuffle(x_train, y_train)

        now = datetime.now().isoformat()

        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae', 'acc'])
        mcp_save = ModelCheckpoint(f'../models/price_ai/{now}.mdl_wts.hdf5', save_best_only=True, monitor='val_loss',
                                   mode='min')
        history = model.fit(
            x_train, y_train, epochs=epochs, validation_data=(x_validate, y_validate), shuffle=True,
            verbose=2, callbacks=[self.tf_board, mcp_save]
        )
        model.save('../models/price_ai__{}'.format(datetime.now().isoformat()))
        return model, history

    def shuffle(self, x_data: np.numarray, y_data: np.numarray):
        """
            Shuffle both arries together
            https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        """
        combined = np.c_[x_data, y_data]
        final_x = combined[:, :, :x_data.shape[2]].reshape(x_data.shape)
        final_y = combined[:, :, -y_data.shape[2]:].reshape(y_data.shape)
        # self.validate_x_y_integrity(final_x, final_y)
        np.random.shuffle(combined)
        return final_x, final_y

    def __split_train_and_test_data(self):
        for symbol, data in self.data_by_symbol.items():
            x_data, y_data = self.shuffle(*self.reshape_data(data))
            test_size = int(self.test_size_percent * len(x_data))
            self.train_data_by_symbol[symbol] = x_data[test_size:], y_data[test_size:]
            self.validate_data_by_symbol[symbol] = x_data[-test_size:], y_data[-test_size:]


class PriceAiModel(BasePriceAi):

    def __init__(self, model_path: str, validation_column='close', look_back=40, forward_prediction=40):
        super().__init__(look_back=look_back, validation_column=validation_column,
                         forward_prediction=forward_prediction)
        self.model: training.Model = keras.models.load_model(model_path)
        self.scaler: MinMaxScaler = MinMaxScaler(copy=False)

    def evaluate(self, data: DataFrame):
        num_items = len(data)
        items_to_drop = int(num_items % self.look_back)

        truncated_data = data[items_to_drop:]
        scaled_data = self.scaler.fit_transform(truncated_data.to_numpy())
        chunked_data = np.array(np.split(scaled_data, len(scaled_data) / self.forward_prediction))

        results = self.model.predict(chunked_data)
        scaled_results = np.array([(r - self.scaler.min_[0]) / self.scaler.scale_[0] for r in results]).flatten()

        prediction_index = truncated_data.index.copy()[self.look_back:]

        df = DataFrame(scaled_results[:-self.forward_prediction], prediction_index, columns=['prediction'])

        return df

    def get_weights(self):
        return self.model.get_weights()
