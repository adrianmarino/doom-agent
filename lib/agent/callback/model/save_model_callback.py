from lib.agent.callback.agent_callback import AgentCallback
from lib.util.os_utils import create_file_path


@AgentCallback.register
class SaveModelCallback:
    def __init__(self, metric_path, freq):
        self.__metric_path = metric_path
        self.__freq = freq

    def perform(self, ctx):
        if ctx.time > 1 and (ctx.is_final_time() or ctx.time % self.__freq == 0):
            ctx.model.save(self.get_path(ctx))

    def get_path(self, ctx):
        return create_file_path(self.__metric_path, f'weights-time_{ctx.time}', 'h5')
