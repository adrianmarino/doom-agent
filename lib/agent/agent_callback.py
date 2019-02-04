import abc


class AgentCallback(abc.ABC):
    @abc.abstractmethod
    def perform(self, agent, time, episode):
        pass
