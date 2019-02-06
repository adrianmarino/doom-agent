from lib.agent.agent_callback import AgentCallback
from lib.metric.tensor_board_metric import TensorBoardMetric


@AgentCallback.register
class TDTargetMetricUpdateCallback:
    def __init__(self, metric_path):
        self.__metric = TensorBoardMetric(metric_path, 'TD Target Update')
        self.repeat = 0

    def perform(self, ctx):
        self.__metric.update(self.__get_value(ctx), ctx.time)

    def __get_value(self, ctx):
        if ctx.is_td_target_update_time():
            self.repeat = 10

        if self.repeat > 0:
            value = 1
            self.repeat -=1
        else:
            value = 0

        return value
