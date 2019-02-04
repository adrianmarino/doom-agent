import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def create_model(
        input_shape,
        action_size,
        learning_rate,
        input_converter
):
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

    return ModelWrapper(model, input_converter)


class FrameWindowToModelInputConverter:
    def convert(self, frames):
        input_array = np.stack((frames), axis=2)
        input_array = np.expand_dims(input_array, axis=0)
        return input_array


class ModelWrapper:
    def __init__(self, model, input_converter):
        self.__model = model
        self.__input_converter = input_converter

    def predict_action_from_frames(self, frames):
        model_input = self.__input_converter.convert(frames)
        actions_q_values = self.__model.predict(model_input)
        best_action_number = np.argmax(actions_q_values)
        return best_action_number

    def predict(self, model_input):
        return self.__model.predict(model_input)

    def copy_weights_to(self, other_model): other_model.set_weights(self.get_weights())

    def save(self, path): self.__model.save_weights(path, overwrite=True)

    def set_weights(self, weights): self.__model.set_weights(weights)

    def get_weights(self): return self.__model.get_weights()

    def fit(self, model_input, model_output, batch_size, callbacks, epochs=1):
        return self.__model.fit(
            model_input,
            model_output,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
