import time

from lib.agent.agent_context import AgentContext
from lib.agent.frame_window import FrameWindow
from lib.agent.phase.agent_phase_factory import AgentPhaseFactory
from lib.agent.phase.impl.agent_final_phase import AgentFinalPhase
from lib.transition.state_transition import StateTransition


class Agent:
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
            train_freq,
            update_target_model_freq,
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

        self.__ctx = AgentContext(
            env,
            model,
            target_model,
            model_train_strategy,
            epsilon,
            logger,
            observe_times,
            explore_times,
            train_freq,
            update_target_model_freq
        )

        self.__phase_factory = AgentPhaseFactory()

    def train(self):
        self.__ctx.reset()
        self.__ctx.env.new_episode()
        phase = self.__phase_factory.create(self.__ctx)

        while not isinstance(phase, AgentFinalPhase):
            if self.__ctx.is_episode_finished():
                self.__new_episode(phase)
                continue
            self.__frame_window.append(self.__current_state_frame())
            initial_state_frames = self.__frame_window.frames()

            action = self.__resolver_action(initial_state_frames)
            rewards = self.__ctx.env.make_action(action)

            if self.__ctx.is_episode_finished():
                self.__new_episode(phase)
                continue
            self.__frame_window.append(self.__current_state_frame())
            final_state_frames = self.__frame_window.frames()

            self.__save_state_transition(action, initial_state_frames, final_state_frames, rewards)

            phase = self.__phase_factory.create(self.__ctx)
            phase.on_each_time(self.__ctx)

            self.__exec_callbacks()
            self.__ctx.increase_time()

    def play(self, episodes, frame_delay, weights_path):
        self.__ctx.model.load(weights_path)
        self.__ctx.reset()
        self.__ctx.env.new_episode()
        phase = self.__phase_factory.create(self.__ctx)

        while self.__ctx.episode < episodes:
            if self.__ctx.is_episode_finished():
                self.__new_episode(phase)
                continue
            self.__frame_window.append(self.__current_state_frame())

            action = self.__ctx.model.predict_action_from_frames(self.__frame_window.frames())
            self.__ctx.env.make_action(action)

            phase = self.__phase_factory.create(self.__ctx)
            phase.on_each_time(self.__ctx)
            time.sleep(frame_delay)


    def __resolver_action(self, initial_state_frames):
        return self.__action_resolver.action(initial_state_frames, self.__ctx.epsilon)

    def __new_episode(self, phase):
        self.__exec_callbacks()
        phase.on_episode_finish(self.__ctx)
        self.__ctx.env.new_episode()
        self.__frame_window.reset()
        self.__ctx.increase_episode()

    def __save_state_transition(self, action, current_frame_window, next_frame_window, rewards):
        state_transition = StateTransition(
            current_frame_window,
            action,
            rewards,
            next_frame_window,
            self.__ctx.is_episode_finished()
        )
        self.__state_transition_memory.add(state_transition)

    def __current_state_frame(self):
        frame = self.__ctx.env.current_state().frame()
        return self.__image_pre_processor.pre_process(frame)

    def __exec_callbacks(self):
        [callback.perform(self.__ctx) for callback in self.__callbacks]
