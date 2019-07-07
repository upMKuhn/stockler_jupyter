from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

look_back = 40
forward_days = 10
num_periods = 20

NUM_NEURONS_FirstLayer = 128
NUM_NEURONS_SecondLayer = 64
EPOCHS = 220

# Build the model
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer, input_shape=(NUM_NEURONS_FirstLayer, 1)))
model.add(Dense(forward_days))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=True,
                    batch_size=2, verbose=2)
