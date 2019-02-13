import time


def str_current_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
