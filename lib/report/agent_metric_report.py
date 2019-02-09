from collections import OrderedDict
from statistics import mean

from lib.metric.tensor_board_metric_reader import TensorBoardMetricReader
from lib.model.model_utils import best_loss


class AgentMetricReport:
    def __init__(self, checkpoint_path, metric_path, metrics, hiperparams, weights_file):
        self.__reader = {}
        for metric in metrics:
            self.__reader[metric] = TensorBoardMetricReader(metric_path, metric)
        self.__checkpoint_path = checkpoint_path
        self.__hiperparams = hiperparams
        self.__weights_file = weights_file

    def __build(self):
        data = OrderedDict()
        data['hiperparams'] = self.__hiperparams
        data['weights'] = self.__weights_file
        data['metrics'] = OrderedDict()
        for name, reader in self.__reader.items():
            values = list(reader.read())
            data['metrics'][name] = {
                'mean': mean(reader.read()),
                'count': len(values),
                'values': sorted([{'value': value, 'occurs': values.count(value)} for value in set(values)],
                                 key=lambda it: it['occurs'], reverse=True)
            }
        data['loss']: best_loss(self.__checkpoint_path)
        return data

    def format_to(self, formatter):
        return formatter.format(self.__build())
