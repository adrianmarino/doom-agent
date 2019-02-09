import glob
import os

import tensorflow as tf


class TensorBoardMetricReader:
    def __init__(self, metric_path, metric_name):
        self._file_paths = glob.glob(os.path.join(metric_path, metric_name, "*"))

    def read(self):
        for file_path in self._file_paths:
            for event in tf.train.summary_iterator(file_path):
                for value in event.summary.value:
                    yield value.simple_value
