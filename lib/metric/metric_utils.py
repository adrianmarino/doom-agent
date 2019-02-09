from lib.metric.avg_calculator import AVG
from lib.metric.tensor_board_metric_reader import TensorBoardMetricReader
from lib.model.model_utils import get_best_weights_file_from, get_loss_model_weights_path


def calculate(metric_path, metrics, calculator):
    values = {}
    for metric in metrics:
        values[f'{metric}_{calculator.name()}'] = TensorBoardMetricReader(metric_path, metric).calculate(calculator)
    return values


def train_metrics_summary(checkpoint_path, metric_path, metric_names):
    values = calculate(metric_path, metric_names, AVG())
    weights_file_path = get_best_weights_file_from(checkpoint_path)
    values['loss'] = get_loss_model_weights_path(weights_file_path)
    values['best_weights'] = weights_file_path
    return values
