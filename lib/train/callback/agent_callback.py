import abc


class AgentCallback(abc.ABC):
    @abc.abstractmethod
    def perform(self, ctx):
        pass
