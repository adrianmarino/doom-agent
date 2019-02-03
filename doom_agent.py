import logging

from keras import backend as K

from lib.action.epsilon_greedy_action_choicer import EpsilonGreedyActionChoicer
from lib.action.epsilon_value import EpsilonValue
from lib.agent.agent import Agent
from lib.environment import Environment
from lib.logger_factory import LoggerFactory
from lib.model.image_pre_processor import ImagePreProcessor
from lib.model.model import create_model
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.train.model_train_strategy import ModelTrainStrategy
from lib.transition.state_transation_memory import StateTransitionMemory
from lib.util.input_shape import InputShape
from lib.util.session_builder import SessionBuilder

# ----------------------------------------------------------------------------------------------------------------------
# Settings...
# ----------------------------------------------------------------------------------------------------------------------
env_config_file = './scenarios/defend_the_center.cfg'

learning_rate = 0.0001
gamma = 0.99
frame_window_size = 4
input_shape = InputShape(rows=64, cols=64, channels=4)

memory_size = 5000
batch_size = 32
time_step_per_train = 100

# Epsilon
epsilon_initial_value = 1.0
epsilon_final_value = 0.001

# Phases
observe_times = 5000
explore_times = 50000

# Frequency
save_model_freq = 1000
copy_weights_to_target_model_freq = 3000

game_state_variables = ['kills', 'ammo', 'health']

logger_config = {
    'logger_name': 'agent',
    'message_format': '%(levelname)s %(asctime)s - %(funcName)s(%(lineno)d) - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'filename': 'logs/agent.log',
    'level': logging.INFO
}


# ----------------------------------------------------------------------------------------------------------------------

def setup_session(): K.set_session(SessionBuilder().regulate_gpu_memory_use().build())


def create_agent():
    logger = LoggerFactory(logger_config).create()

    env = Environment(
        config_file=env_config_file,
        advance_steps=input_shape.channels,
        rewards_computation_strategy=DoomRewardsComputationStrategy(),
        variable_names=game_state_variables
    )

    model = create_model(input_shape, len(env.current_state().variables), learning_rate)
    target_model = create_model(input_shape, len(env.current_state().variables), learning_rate)

    epsilon = EpsilonValue(epsilon_initial_value, epsilon_final_value, observe_times, explore_times)
    action_choicer = EpsilonGreedyActionChoicer(action_size=env.actions_count(), model=model, epsilon=epsilon)

    state_transition_memory = StateTransitionMemory(memory_size)

    model_train_strategy = ModelTrainStrategy(
        model,
        target_model,
        state_transition_memory,
        batch_size,
        time_step_per_train,
        input_shape,
        gamma
    )

    image_pre_processor = ImagePreProcessor((input_shape.rows, input_shape.cols))

    return Agent(
        env,
        input_shape,
        model,
        target_model,
        model_train_strategy,
        epsilon,
        action_choicer,
        state_transition_memory,
        image_pre_processor,
        logger,
        observe_times,
        explore_times,
        save_model_freq,
        time_step_per_train,
        copy_weights_to_target_model_freq
    )


if __name__ == "__main__":
    setup_session()
    agent = create_agent()
    agent.train()
