from lib.train.model.callback.checkpoint_callback_factory import CheckpointCallbackFactory
from lib.train.model.callback.tensor_board_callback_factory import TensorBoardCallbackFactory


class ModelCallbackFactory:
    def __init__(self, cfg):
        self.__cfg = cfg

    def create_all(self, names):
        return [self.create(name) for name in names]

    def create(self, name):
        if 'tensor_board' == name:
            return TensorBoardCallbackFactory.create(
                self.__cfg['callbacks.model.settings.tensor_board.metric_path'],
                self.__cfg['callbacks.model.settings.tensor_board.batch_size']
            )
        elif 'checkpoint' == name:
            return CheckpointCallbackFactory.create(
                self.__cfg['callbacks.model.settings.checkpoint.path'],
                self.__cfg['callbacks.model.settings.checkpoint.monitor']
            )
        else:
            raise Exception(f'Not found {name} callback')
