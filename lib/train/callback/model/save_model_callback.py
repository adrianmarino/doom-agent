from lib.train.callback.algorithm_callback import AlgorithmCallback
from lib.util.os_utils import create_file_path


@AlgorithmCallback.register
class SaveModelCallback:
    def __init__(self, metric_path, freq):
        self.__metric_path = metric_path
        self.__freq = freq

    def perform(self, ctx):
        if self.is_time_to_save(ctx):
            ctx.model.save(self.get_path(ctx))

    def is_time_to_save(self, ctx):
        return ctx.time > 0 and ctx.time % self.freq(ctx) == 0

    def get_path(self, ctx):
        return create_file_path(self.__metric_path, f'weights-time_{ctx.time}-loss_0', 'h5')

    def freq(self, ctx):
        return ctx.final_time() if self.__freq == 0 else self.__freq
