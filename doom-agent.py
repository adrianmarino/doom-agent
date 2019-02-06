from keras import backend as K

from lib.action.epsilon_greedy_action_choicer import EpsilonGreedyActionResolver
from lib.action.epsilon_value import EpsilonValue
from lib.agent.agent import Agent
from lib.agent.callback.agent_callback_factory import AgentCallbackFactory
from lib.env.environment import Environment
from lib.logger_factory import LoggerFactory
from lib.metric.tensor_board_callback_factory import TensorBoardCallbackFactory
from lib.model.image_pre_processor import ImagePreProcessor
from lib.model.model import create_model, FrameWindowToModelInputConverter
from lib.model.model_train_strategy import ModelTrainStrategy
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.transition.state_transation_memory import StateTransitionMemory
from lib.util.config import Config
from lib.util.input_shape import InputShape
from lib.util.session_builder import SessionBuilder


def setup_session():
    K.set_session(SessionBuilder().regulate_gpu_memory_use().build())


def create_agent(cfg):
    logger = LoggerFactory(cfg['logger']).create()

    input_shape = InputShape.from_str(cfg['net.input_shape'])

    env = Environment(
        config_file=cfg['env.config_file'],
        advance_steps=input_shape.channels,
        rewards_computation_strategy=DoomRewardsComputationStrategy(),
        variable_names=cfg['env.variables'],
        window_visible=cfg['env.show']
    )

    input_converter = FrameWindowToModelInputConverter()
    model = create_model(input_shape, env.actions_count(), cfg['train.lr'], input_converter)
    target_model = create_model(input_shape, env.actions_count(), cfg['train.lr'], input_converter)

    epsilon = EpsilonValue(cfg['epsilon.initial'], cfg['epsilon.final'], cfg['phase_time.explore'])

    action_resolver = EpsilonGreedyActionResolver(model, env.actions_count(), epsilon)

    state_transition_memory = StateTransitionMemory(cfg['memory_size'])

    model_train_callbacks = [TensorBoardCallbackFactory.create(cfg['metric.path'], cfg['train.batch_size'])]

    model_train_strategy = ModelTrainStrategy(
        model,
        target_model,
        state_transition_memory,
        cfg['train.batch_size'],
        cfg['train.freq'],
        input_shape,
        cfg['train.gamma'],
        input_converter,
        model_train_callbacks
    )

    image_pre_processor = ImagePreProcessor((input_shape.rows, input_shape.cols))

    agent_callbacks = AgentCallbackFactory(cfg).create_all(
        ['epsilon', 'td_target_update', 'kills', 'ammo', 'health', 'save_model']
    )

    return Agent(
        env,
        input_shape,
        model,
        target_model,
        model_train_strategy,
        epsilon,
        action_resolver,
        state_transition_memory,
        image_pre_processor,
        logger,
        cfg['phase_time.observe'],
        cfg['phase_time.explore'],
        cfg['train.freq'],
        cfg['train.update_target_model_freq'],
        agent_callbacks
    )


if __name__ == "__main__":
    setup_session()
    config = Config('./config.yml')
    agent = create_agent(config)
    agent.train()
