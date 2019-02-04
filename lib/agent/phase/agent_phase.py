import abc


class AgentPhase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, agent):
        pass

    @abc.abstractmethod
    def perform(self, time, episode):
        pass
