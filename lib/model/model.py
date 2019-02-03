from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
import numpy as np


def create_model(input_shape, action_size, learning_rate):
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

    return ModelWrapper(model)


class ModelWrapper:
    def __init__(self, model): self.__model = model

    def predict_action(self, frames):
        model_input = self.__to_input(frames)
        actions_q_values = self.__model.predict(model_input)
        best_action_number = np.argmax(actions_q_values)
        return best_action_number

    def copy_weights_to(self, other_model): other_model.set_weights(self.get_weights())

    def save(self, path): self.__model.save_weights(path, overwrite=True)

    def set_weights(self, weights): self.__model.set_weights(weights)

    def get_weights(self): return self.__model.get_weights()

    def __to_input(self, frames):
        input_array = np.stack((frames), axis=2)
        input_array = np.expand_dims(input_array, axis=0)
        return input_array
