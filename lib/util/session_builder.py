import tensorflow as tf


class SessionBuilder:

    def __init__(self): self.config = tf.ConfigProto()

    def regulate_gpu_memory_use(self):
        self.config.gpu_options.allow_growth=True
        return self

    def build(self): return tf.Session(config=self.config)
