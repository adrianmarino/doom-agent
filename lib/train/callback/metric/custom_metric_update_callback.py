from lib.train.callback.agent_callback import AgentCallback
from lib.train.metric.tensor_board_metric_writer import TensorBoardMetricWriter


@AgentCallback.register
class CustomMetricUpdateCallback:
    def __init__(self, metric_path, metric_name, value_resolver, display_name='', description='', each_episode=False):
        self.__metric = TensorBoardMetricWriter(metric_path, metric_name, display_name, description)
        self.__value_resolver = value_resolver
        self.__each_episode = each_episode

    def perform(self, ctx):
        if self.__each_episode:
            if ctx.env.is_episode_finished():
                self.__metric.write(self.__value_resolver(ctx), ctx.time)
        else:
            self.__metric.write(self.__value_resolver(ctx), ctx.time)
