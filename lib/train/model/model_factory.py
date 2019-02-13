from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

from lib.train.model.model_wrapper import ModelWrapper


class ModelFactory:
    def __init__(self, input_converter, logger):
        self.__input_converter = input_converter
        self.__logger = logger

    def create(self, name, input_shape, action_size, learning_rate):
        if 'model_a' == name:
            return self.__create_model_a(input_shape, action_size, learning_rate)
        elif 'model_b' == name:
            return self.__create_model_b(input_shape, action_size, learning_rate)
        else:
            raise Exception(f'Not found {name} model')

    def __create_model_a(self, input_shape, action_size, learning_rate):
        model = Sequential()

        model.add(Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape.as_tuple()
        ))
        model.add(Conv2D(filters=64, kernel_size=[4, 4], strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1), activation='relu'))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_size, activation='linear'))

        optimizer = Adam(lr=learning_rate)

        model.compile(loss='mse', optimizer=optimizer)

        return ModelWrapper(model, self.__input_converter, self.__logger)

    def __create_model_b(self, input_shape, action_size, learning_rate):
        model = Sequential()

        model.add(Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            activation='relu',
            input_shape=input_shape.as_tuple()
        ))
        model.add(Conv2D(filters=64, kernel_size=[4, 4], strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1), activation='relu'))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(512, activation='relu'))

        model.add(Dense(100, activation='relu'))

        model.add(Dense(action_size, activation='linear'))

        optimizer = Adam(lr=learning_rate)

        model.compile(loss='mse', optimizer=optimizer)

        return ModelWrapper(model, self.__input_converter, self.__logger)
