import tensorflow as tf

class SummaryUtils:
   @staticmethod
   def create_value(name, value):
      summary = tf.Summary()
      summary.value.add(tag=name, simple_value=value)

class TensorBoardMetric:
   def __init__(self, path, name):
      self.name = name
      self.__writer = tf.summary.FileWriter(path)

   def update(self, value):
      self.__writer.add_summary(SummaryUtils.create_value(self.name, value))
      self.__writer.flush()