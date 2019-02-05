from lib.agent.agent_callback import AgentCallback
from lib.metric.tensor_board_metric import TensorBoardMetric


@AgentCallback.register
class TDTargetMetricUpdateCallback:
    def __init__(self, metric_path):
        self.__metric = TensorBoardMetric(metric_path, 'TD Target Update')
        self.repeat = 0

    def perform(self, agent, time, episode):
        value = self.__get_value(agent, time)
        self.__metric.update(value, time)

    def __get_value(self, agent, time):
        if time % agent.update_target_model_freq == 0:
            self.repeat = 10

        if self.repeat > 0:
            value = 1
            self.repeat -=1
        else:
            value = 0

        return value


