import abc


class AlgorithmCallback(abc.ABC):
    @abc.abstractmethod
    def perform(self, ctx):
        pass
