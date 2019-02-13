from collections import OrderedDict
from statistics import mean

from lib.train.metric.tensor_board_metric_reader import TensorBoardMetricReader
from lib.train.model.model_utils import best_loss


class AgentMetricReport:
    def __init__(self, checkpoint_path, metric_path, metrics, config, weights_file, last_times=50):
        self.__reader = {}
        for metric in metrics:
            self.__reader[metric] = TensorBoardMetricReader(metric_path, metric)
        self.__checkpoint_path = checkpoint_path
        self.__config = config
        self.__weights_file = weights_file
        self.__last_times = last_times

    def __build(self):
        data = OrderedDict()
        data['config'] = self.__config
        data['weights'] = self.__weights_file
        data['metrics'] = OrderedDict()
        for name, reader in self.__reader.items():

            values = list(reader.read())
            if self.__last_times > 0:
                values = values[:-self.__last_times]

            data['metrics'][name] = {
                'mean': mean(values),
                'count': len(values),
                'values': sorted([{'occurs': values.count(value), 'value': value} for value in set(values)],
                                 key=lambda it: it['occurs'], reverse=True)
            }
        data['loss']: best_loss(self.__checkpoint_path)
        return data

    def format_to(self, formatter):
        return formatter.format(self.__build())
