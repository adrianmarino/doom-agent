import os

from keras.callbacks import ModelCheckpoint

from lib.util.os_utils import create_path


class CheckpointCallbackFactory:
    @staticmethod
    def create(path, monitor='loss'):
        return ModelCheckpoint(
            os.path.join(create_path(path), 'weights__loss_{loss:.4f}.h5'),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
