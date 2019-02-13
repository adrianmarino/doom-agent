from lib.env.environment import Environment
from lib.train.action.epsilon_greedy_action_choicer import EpsilonGreedyActionResolver
from lib.train.action.epsilon_value import EpsilonValue
from lib.train.algorithm.ddqn.ddqn_train_algorithm import DDQNTrainAlgorithm
from lib.train.callback.algorithm_callback_factory import AlgorithmCallbackFactory
from lib.train.model.callback.model_callback_factory import ModelCallbackFactory
from lib.train.model.frame_window_to_model_input_converter import FrameWindowToModelInputConverter
from lib.train.model.image_pre_processor import ImagePreProcessor
from lib.train.model.model_factory import ModelFactory
from lib.train.model.model_fit_strategy import ModelFitStrategy
from lib.train.model.td_target_update_freq_resolver import TDTargetUpdateFreqResolver
from lib.train.transition.state_transation_memory import StateTransitionMemory
from lib.util.input_shape import InputShape


class DDQNTrainAlgorithmFactory:
    def __init__(self, logger):
        self.__logger = logger

    def create(self, cfg, rewards_computation_strategy):
        env = Environment(
            config_file=cfg['env.config_file'],
            advance_steps=cfg['env.train.advance_steps'],
            rewards_computation_strategy=rewards_computation_strategy,
            variable_names=cfg['env.variables'],
            window_visible=cfg['env.train.show'],
            sound_enabled=cfg['env.train.sound']
        )

        input_shape = InputShape.from_str(cfg['hiperparams.input_shape'])

        input_converter = FrameWindowToModelInputConverter()

        model_factory = ModelFactory(input_converter, self.__logger)

        model = model_factory.create(
            cfg['hiperparams.model'],
            input_shape,
            env.actions_count(),
            cfg['hiperparams.lr']
        )
        target_model = model_factory.create(
            cfg['hiperparams.model'],
            input_shape,
            env.actions_count(),
            cfg['hiperparams.lr']
        )

        epsilon = EpsilonValue(
            cfg['hiperparams.epsilon.initial'],
            cfg['hiperparams.epsilon.final'],
            cfg['hiperparams.phase_time.explore']
        )

        action_resolver = EpsilonGreedyActionResolver(model, env.actions_count(), epsilon)

        state_transition_memory = StateTransitionMemory(cfg['hiperparams.memory_size'])

        model_callbacks = ModelCallbackFactory(cfg).create_all(cfg['callbacks.model.active'])

        model_train_strategy = ModelFitStrategy(
            model,
            target_model,
            state_transition_memory,
            cfg['hiperparams.batch_size'],
            cfg['hiperparams.train_freq'],
            input_shape,
            cfg['hiperparams.gamma'],
            input_converter,
            model_callbacks
        )

        image_pre_processor = ImagePreProcessor(
            input_shape.rows,
            input_shape.cols,
            cfg['hiperparams.chop_bottom_height']
        )

        algorithm_callbacks = AlgorithmCallbackFactory(cfg).create_all(cfg['callbacks.algorithm.active'])

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
            self.__logger,
            cfg['hiperparams.phase_time.observe'],
            cfg['hiperparams.phase_time.explore'],
            cfg['hiperparams.phase_time.train'],
            cfg['hiperparams.train_freq'],
            td_target_update_freq_resolver,
            algorithm_callbacks
        )
