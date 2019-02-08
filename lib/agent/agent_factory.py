from lib.action.epsilon_greedy_action_choicer import EpsilonGreedyActionResolver
from lib.action.epsilon_value import EpsilonValue
from lib.agent.agent import Agent
from lib.agent.callback.agent_callback_factory import AgentCallbackFactory
from lib.logger_factory import LoggerFactory
from lib.metric.tensor_board_callback_factory import TensorBoardCallbackFactory
from lib.model.image_pre_processor import ImagePreProcessor
from lib.model.model import FrameWindowToModelInputConverter, create_model
from lib.model.model_train_strategy import ModelTrainStrategy
from lib.transition.state_transation_memory import StateTransitionMemory
from lib.util.input_shape import InputShape


class AgentFactory:

    @staticmethod
    def create(cfg, env):
        logger = LoggerFactory(cfg['logger']).create()

        input_shape = InputShape.from_str(cfg['hiperparams.input_shape'])

        input_converter = FrameWindowToModelInputConverter()
        model = create_model(input_shape, env.actions_count(), cfg['hiperparams.lr'], input_converter, logger)
        target_model = create_model(input_shape, env.actions_count(), cfg['hiperparams.lr'], input_converter, logger)

        epsilon = EpsilonValue(
            cfg['hiperparams.epsilon.initial'],
            cfg['hiperparams.epsilon.final'],
            cfg['hiperparams.phase_time.explore']
        )

        action_resolver = EpsilonGreedyActionResolver(model, env.actions_count(), epsilon)

        state_transition_memory = StateTransitionMemory(cfg['hiperparams.memory_size'])

        model_train_callbacks = [
            TensorBoardCallbackFactory.create(cfg['metric.path'], cfg['hiperparams.batch_size'])
        ]

        model_train_strategy = ModelTrainStrategy(
            model,
            target_model,
            state_transition_memory,
            cfg['hiperparams.batch_size'],
            cfg['hiperparams.train_freq'],
            input_shape,
            cfg['hiperparams.gamma'],
            input_converter,
            model_train_callbacks
        )

        image_pre_processor = ImagePreProcessor(
            input_shape.rows,
            input_shape.cols,
            cfg['hiperparams.chop_bottom_height']
        )

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
            cfg['hiperparams.phase_time.observe'],
            cfg['hiperparams.phase_time.explore'],
            cfg['hiperparams.train_freq'],
            cfg['hiperparams.update_target_model_freq'],
            agent_callbacks
        )