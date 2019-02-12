from lib.train.algorithm.ddqn.ddqn_algorithm_context import DDQNAlgorithmContext
from lib.train.algorithm.ddqn.phase.impl.algorithm_final_phase import AlgorithmFinalPhase
from lib.train.algorithm.ddqn.phase.impl.algorithm_phase_factory import AlgorithmPhaseFactory
from lib.train.algorithm.train_algoritm import TrainAlgorithm
from lib.train.frame_window import FrameWindow
from lib.train.transition.state_transition import StateTransition


@TrainAlgorithm.register
class DDQNTrainAlgorithm:
    def __init__(
            self,
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
            observe_times,
            explore_times,
            train_times,
            train_freq,
            td_target_update_freq_resolver,
            callbacks=()
    ):
        self.__action_resolver = action_resolver
        self.__state_transition_memory = state_transition_memory
        self.__image_pre_processor = image_pre_processor
        self.__callbacks = callbacks

        self.__frame_window = FrameWindow(
            frame_shape=(input_shape.rows, input_shape.cols),
            size=input_shape.channels
        )

        self.__ctx = DDQNAlgorithmContext(
            env,
            model,
            target_model,
            model_train_strategy,
            epsilon,
            logger,
            observe_times,
            explore_times,
            train_times,
            train_freq,
            td_target_update_freq_resolver
        )
        self.__phase_factory = AlgorithmPhaseFactory()

    def train(self, weights_path):
        self.__ctx.model.load(weights_path)
        self.__ctx.reset()
        self.__ctx.env.new_episode()
        phase = self.__phase_factory.create(self.__ctx)

        while not isinstance(phase, AlgorithmFinalPhase):
            if self.__ctx.env.is_episode_finished():
                self.__new_episode(phase)
                continue
            self.__frame_window.append(self.__current_state_frame())
            initial_state_frames = self.__frame_window.frames()

            action = self.__action_resolver.action(initial_state_frames, self.__ctx.epsilon)
            rewards = self.__ctx.env.make_action(action)

            if self.__ctx.env.is_episode_finished():
                self.__new_episode(phase)
                continue
            self.__frame_window.append(self.__current_state_frame())
            final_state_frames = self.__frame_window.frames()

            self.__state_transition_memory.add(
                StateTransition(
                    initial_state_frames,
                    action,
                    rewards,
                    final_state_frames,
                    self.__ctx.env.is_episode_finished()
                )
            )

            phase = self.__phase_factory.create(self.__ctx)
            phase.on_each_time(self.__ctx)

            self.__exec_callbacks()
            self.__ctx.increase_time()

    def __new_episode(self, phase):
        self.__exec_callbacks()
        phase.on_episode_finish(self.__ctx)
        self.__ctx.env.new_episode()
        self.__frame_window.reset()
        self.__ctx.increase_episode()

    def __current_state_frame(self):
        frame = self.__ctx.env.current_state().frame()
        return self.__image_pre_processor.pre_process(frame)

    def __exec_callbacks(self):
        [callback.perform(self.__ctx) for callback in self.__callbacks]
