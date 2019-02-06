import os

from lib.agent.callback.agent_callback import AgentCallback
from lib.util.os_utils import create_directory


@AgentCallback.register
class SaveModelCallback:
    def __init__(self, path, freq):
        self.__path = path
        self.__freq = freq
        create_directory(path)

    def perform(self, ctx):
        if ctx.time > 1 and (ctx.is_final_time() or ctx.time % self.__freq == 0):
            ctx.model.save(self.get_path(ctx))

    def get_path(self, ctx):
        return os.path.join(self.__path, f'weights-time_{ctx.time}.h5')
