import abc


class AlgorithmPhase(abc.ABC):
    @abc.abstractmethod
    def on_each_time(self, ctx):
        pass

    @abc.abstractmethod
    def on_episode_finish(self, ctx):
        pass
