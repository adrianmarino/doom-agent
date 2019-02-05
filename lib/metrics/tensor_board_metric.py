import os
import tensorflow as tf

from lib.util.os_utils import create_directory


class SummaryUtils:
    @staticmethod
    def scalar(name, value):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        return summary


class TensorBoardMetric:
    def __init__(self, path, name):
        self.name = name
        create_directory(path)
        self.__writer = tf.summary.FileWriter(os.path.join(path, name))

    def update(self, value, time):
        summary = SummaryUtils.scalar(self.name, value)
        self.__writer.add_summary(summary, time)
        self.__writer.flush()
