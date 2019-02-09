import tensorflow as tf


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
