import abc


class TrainAlgorithm(abc.ABC):
    @abc.abstractmethod
    def train(self, weights_path):
        pass
