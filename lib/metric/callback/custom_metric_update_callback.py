from lib.agent.agent_callback import AgentCallback
from lib.metric.tensor_board_metric import TensorBoardMetric


@AgentCallback.register
class CustomMetricUpdateCallback:
    def __init__(self, path, label, value_resolver, each_episode=False):
        self.__metric = TensorBoardMetric(path, label)
        self.__value_resolver = value_resolver
        self.__each_episode = each_episode

    def perform(self, agent, time, episode):
        if self.__each_episode:
            if agent.env.is_episode_finished():
                self.__metric.update(self.__value_resolver(agent, time, episode), time)
        else:
            self.__metric.update(self.__value_resolver(agent, time, episode), time)

