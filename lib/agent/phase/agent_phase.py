import abc


class AgentPhase(abc.ABC):
    @abc.abstractmethod
    def on_each_time(self, ctx):
        pass

    @abc.abstractmethod
    def on_episode_finish(self, ctx):
        pass
