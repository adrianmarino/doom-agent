from keras import backend as K

from lib.frame_window import FrameWindow
from lib.input_shape import InputShape
from lib.memory import Memory
from lib.model import create_model
from lib.session_builder import SessionBuilder
from lib.environment import Environment
from lib.epsilon_greedy_action_choicer import EpsilonGreedyActionChoicer


def setup_session(): K.set_session(SessionBuilder().regulate_gpu_memory_use().build())


class RLAgent:
    def __init__(self, env, action_choicer, model):
        self.__env = env
        self.__action_choicer = action_choicer
        self.__model = model


if __name__ == "__main__":
    learning_rate = 0.0001
    frame_window_size = 4
    input_shape = InputShape(rows=64, cols=64, channels=4)
    memory_size = 5000

    setup_session()

    env = Environment(
        config_file='./scenarios/defend_the_center.cfg',
        advance_steps=input_shape.channels
    )
    env.new_episode()

    model = create_model(input_shape.as_tuple(), env.possible_actions_size(), learning_rate)
    model_target = create_model(input_shape.as_tuple(), env.possible_actions_size(), learning_rate)

    action_choicer = EpsilonGreedyActionChoicer(action_size=env.possible_actions_size(), model=model)

    agent = RLAgent(env, action_choicer, model)

    frame_window = FrameWindow(frame_shape=(input_shape.rows, input_shape.cols), size=input_shape.channels)

    memory = Memory(memory_size)

    while not env.is_episode_finished():
        frame_window.append(env.current_state())

        action = action_choicer.choice_action(model, env.current_state())

        check_point = env.make_action(action)
