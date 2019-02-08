from keras import backend as K

from lib.util.session_builder import SessionBuilder


def setup_session():
    K.set_session(SessionBuilder().regulate_gpu_memory_use().build())
