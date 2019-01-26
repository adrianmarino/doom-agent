from keras.models import Sequencial
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def create_model(input_shape, action_size, learning_rate):
    model = Sequencial()
    model.add(Conv2D(filters=32, kernel_size=[8, 8], strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=[4, 4], strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model
