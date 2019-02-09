from lib.metric.summary_builder import SummaryBuilder
from lib.util.os_utils import create_file_path
import tensorflow as tf

class TensorBoardMetricWriter:
    def __init__(self, metric_path, metric_name, display_name='', description=''):
        self.__metric_name = metric_name
        self.__writer = tf.summary.FileWriter(create_file_path(metric_path, metric_name))
        self.__metadata = SummaryBuilder.metadata(display_name, description)

    def write(self, value, time):
        summary = SummaryBuilder.scalar(
            metric_name=self.__metric_name,
            value=value,
            metadata=self.__metadata
        )
        self.__writer.add_summary(summary, time)
        self.__writer.flush()
