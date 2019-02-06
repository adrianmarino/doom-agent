import tensorflow as tf

from lib.util.os_utils import create_file_path


class SummaryBuilder:
    @staticmethod
    def metadata(display_name, description):
        metadata = tf.SummaryMetadata()
        metadata.display_name = display_name
        metadata.summary_description = description
        return metadata

    @staticmethod
    def scalar(metric_name, value, metadata=None):
        summary = tf.Summary()
        summary.value.add(
            tag=metric_name,
            simple_value=value,
            metadata=metadata
        )
        return summary


class TensorBoardMetric:
    def __init__(self, metric_path, metric_name, display_name='', description=''):
        self.__metric_name = metric_name
        self.__writer = tf.summary.FileWriter(create_file_path(metric_path, metric_name))
        self.__metadata = SummaryBuilder.metadata(display_name, description)

    def update(self, value, time):
        summary = SummaryBuilder.scalar(
            metric_name=self.__metric_name,
            value=value,
            metadata=self.__metadata
        )
        self.__writer.add_summary(summary, time)
        self.__writer.flush()
