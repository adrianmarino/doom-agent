from lib.env.environment import Environment
from lib.logger_factory import LoggerFactory
from lib.train.action.epsilon_greedy_action_choicer import EpsilonGreedyActionResolver
from lib.train.action.epsilon_value import EpsilonValue
from lib.train.algorithm.ddqn.ddqn_train_algorithm import DDQNTrainAlgorithm
from lib.train.callback.agent_callback_factory import AgentCallbackFactory
from lib.train.model.callback.checkpoint_callback_factory import CheckpointCallbackFactory
from lib.train.model.callback.tensor_board_callback_factory import TensorBoardCallbackFactory
from lib.train.model.image_pre_processor import ImagePreProcessor
from lib.train.model.model import FrameWindowToModelInputConverter, create_model
from lib.train.model.model_fit_strategy import ModelFitStrategy
from lib.train.model.td_target_update_freq_resolver import TDTargetUpdateFreqResolver
from lib.train.transition.state_transation_memory import StateTransitionMemory
from lib.util.input_shape import InputShape


class DDQNTrainAlgorithmFactory:

    @staticmethod
    def create(cfg, rewards_computation_strategy):
        env = Environment(
            config_file=cfg['env.config_file'],
            advance_steps=cfg['env.train.advance_steps'],
            rewards_computation_strategy=rewards_computation_strategy,
            variable_names=cfg['env.variables'],
            window_visible=cfg['env.train.show'],
            sound_enabled=cfg['env.train.sound']
        )

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
            TensorBoardCallbackFactory.create(cfg['metric.path'], cfg['hiperparams.batch_size']),
            CheckpointCallbackFactory.create(cfg['checkpoint.path'], 'loss')
        ]

        model_train_strategy = ModelFitStrategy(
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
            ['epsilon', 'td_target_update', 'kills', 'ammo', 'health']  # , 'save_model']
        )

        td_target_update_freq_resolver = TDTargetUpdateFreqResolver(
            cfg['hiperparams.update_target_model_freq_schedule']
        )

        return DDQNTrainAlgorithm(
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
            cfg['hiperparams.phase_time.train'],
            cfg['hiperparams.train_freq'],
            td_target_update_freq_resolver,
            agent_callbacks
        )
