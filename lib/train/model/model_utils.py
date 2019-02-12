import re

from lib.util.os_utils import min_file_path_from


def best_loss(checkpoint_path):
    best_weights_file = get_best_weights_file_from(checkpoint_path)
    return get_loss_model_weights_path(best_weights_file)

def get_loss_model_weights_path(file_path):
    loss = re.search("loss_(.*).h5", file_path).group(1)
    return float(loss)


def get_best_weights_file_from(path):
    return min_file_path_from(f'{path}/*.h5', get_loss_model_weights_path)
