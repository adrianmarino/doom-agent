import re

from lib.util.os_utils import min_file_path_from


def get_loss_model_weights_path(file_path):
    loss = re.search("loss_(.*).h5", file_path).group(1)
    return float(loss)


def get_best_weights_file_from(path):
    return min_file_path_from(f'{path}/*.h5', get_loss_model_weights_path)
