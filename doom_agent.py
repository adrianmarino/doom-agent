from keras import backend as K

from agent import Agent
from lib.environment import Environment
from lib.epsilon_greedy_action_choicer import EpsilonGreedyActionChoicer
from lib.epsilon_value import EpsilonValue
from lib.experience_memory import StateTransitionMemory
from lib.frame_window import FrameWindow
from lib.input_shape import InputShape
from lib.model import create_model
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.session_builder import SessionBuilder
from lib.train.model_train_strategy import ModelTrainStrategy

# ----------------------------------------------------------------------------------------------------------------------
# Settings...
# ----------------------------------------------------------------------------------------------------------------------
learning_rate = 0.0001
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
# ----------------------------------------------------------------------------------------------------------------------


def setup_session(): K.set_session(SessionBuilder().regulate_gpu_memory_use().build())

def create_agent():
    env = Environment(
        config_file='./scenarios/defend_the_center.cfg',
        advance_steps=input_shape.channels,
        rewards_computation_strategy=DoomRewardsComputationStrategy()
    )

    model = create_model(input_shape, env.possible_actions_size(), learning_rate)
    target_model = create_model(input_shape, env.possible_actions_size(), learning_rate)

    epsilon = EpsilonValue(epsilon_initial_value, epsilon_final_value, observe_times, explore_times)
    action_choicer = EpsilonGreedyActionChoicer(action_size=env.possible_actions_size(), model=model, epsilon=epsilon)

    state_transition_memory = StateTransitionMemory(memory_size)

    model_train_strategy = ModelTrainStrategy(
        model, target_model, state_transition_memory, batch_size, time_step_per_train
    )

    return Agent(
        env,
        model,
        target_model,
        model_train_strategy,
        epsilon,
        action_choicer,
        state_transition_memory,
        observe_times,
        explore_times,
        save_model_freq,
        copy_weights_to_target_model_freq
    )


if __name__ == "__main__":
    setup_session()
    agent = create_agent()
    agent.train()
