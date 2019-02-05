import os

from keras.callbacks import ModelCheckpoint

from lib.util.os_utils import create_directory

class CheckpointFactory:
    @staticmethod
    def create(path, monitor='loss'):
        create_directory(path)
        return ModelCheckpoint(
            os.path.join(path, 'weights__loss_{loss:.4f}.h5'),
            monitor=monitor,
            verbose=1,
            save_best_only=False,
            mode='auto'
        )

