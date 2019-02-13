import numpy as np


class ModelWrapper:
    def __init__(self, model, input_converter, logger):
        self.__model = model
        self.__input_converter = input_converter
        self.__logger = logger

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

    def load(self, path):
        if path:
            self.__logger.info(f'Load weights from {path}')
            self.__model.load_weights(path)
