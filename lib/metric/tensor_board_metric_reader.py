import glob
import os

import tensorflow as tf


class TensorBoardMetricReader:
    def __init__(self, metric_path, metric_name):
        self._file_paths = glob.glob(os.path.join(metric_path, metric_name, "*"))

    def read(self, read_value):
        for file_path in self._file_paths:
            for event in tf.train.summary_iterator(file_path):
                for value in event.summary.value:
                    read_value(value)

    def calculate(self, calculator):
        self.read(lambda value: calculator.next(value.simple_value))
        return calculator.result()
