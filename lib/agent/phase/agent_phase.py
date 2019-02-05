import abc


class AgentPhase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, agent):
        pass

    @abc.abstractmethod
    def each_time(self, time, episode):
        pass

    @abc.abstractmethod
    def on_episode_finish(self, time, episode):
        pass
