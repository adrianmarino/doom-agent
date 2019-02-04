from keras.callbacks import TensorBoard


class MetricsBoardFactory:
    @staticmethod
    def create(logs_path, batch_size):
        return TensorBoard(
            log_dir=logs_path,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            batch_size=batch_size,
            write_grads=True,
            update_freq='batch'
        )
