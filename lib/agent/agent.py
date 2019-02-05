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
        self.env = env
        self.model = model
        self.target_model = target_model
        self.model_train_strategy = model_train_strategy
        self.epsilon = epsilon
        self.action_resolver = action_resolver
        self.state_transition_memory = state_transition_memory
        self.image_pre_processor = image_pre_processor
        self.frame_window = FrameWindow(
            frame_shape=(input_shape.rows, input_shape.cols),
            size=input_shape.channels
        )
        self.logger = logger
        self.train_freq = train_freq
        self.update_target_model_freq = update_target_model_freq

        self.observe_times = observe_times
        self.explore_times = explore_times
        self.__callbacks = callbacks
        self.__phase_factory = AgentPhaseFactory(
            self,
            self.observe_times,
            self.explore_times,
            self.train_freq,
            self.update_target_model_freq
        )

    def train(self):
        time = 0
        episode = self.__new_episode(0)
        phase = self.__phase_factory.create(self, time)

        while not isinstance(phase, AgentFinalPhase):
            if self.env.is_episode_finished():
                episode = self.__new_episode(episode)
                continue
            self.frame_window.append(self.__current_state_frame())
            initial_state_frames = self.frame_window.frames()

            action = self.action_resolver.action(initial_state_frames, self.epsilon)

            rewards = self.env.make_action(action)

            if self.env.is_episode_finished():
                episode = self.__new_episode(episode)
                continue
            self.frame_window.append(self.__current_state_frame())
            final_state_frames = self.frame_window.frames()

            self.__save_state_transition(action, initial_state_frames, final_state_frames, rewards)

            phase = self.__phase_factory.create(self, time)
            phase.perform(time, episode)

            time += 1

    def __new_episode(self, episode):
        self.env.new_episode()
        self.frame_window.reset()
        return episode + 1

    def __save_state_transition(self, action, current_frame_window, next_frame_window, rewards):
        state_transition = StateTransition(
            current_frame_window,
            action,
            rewards,
            next_frame_window,
            self.env.is_episode_finished()
        )
        self.state_transition_memory.add(state_transition)

    def __current_state_frame(self):
        frame = self.env.current_state().frame()
        return self.image_pre_processor.pre_process(frame)

    def exec_callbacks(self, time, episode):
        [callback.perform(self, time, episode) for callback in self.__callbacks]
