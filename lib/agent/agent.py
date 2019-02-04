from enum import Enum

from lib.agent.frame_window import FrameWindow
from lib.transition.state_transition import StateTransition


class AgentPhase(Enum):
    OBSERVATION = 1,
    EXPLORATION = 2,
    TRAIN = 3,
    FINAL = 4,
    UNKNOWN = 5


class Agent:
    def __init__(
            self,
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
            observe_times=5000,
            explore_times=50000,
            full_train_times=0,
            time_step_per_train=100,
            copy_weights_to_target_model_freq=3000
    ):
        self.__env = env
        self.__model = model
        self.__target_model = target_model
        self.__model_train_strategy = model_train_strategy
        self.__epsilon = epsilon
        self.__action_choicer = action_choicer
        self.__state_transition_memory = state_transition_memory
        self.__image_pre_processor = image_pre_processor
        self.__frame_window = FrameWindow(frame_shape=(input_shape.rows, input_shape.cols), size=input_shape.channels)
        self.__logger = logger

        self.__observe_times = observe_times
        self.__explore_times = explore_times
        self.__full_train_times = full_train_times
        self.__time_step_per_train = time_step_per_train
        self.__copy_weights_to_target_model_freq = copy_weights_to_target_model_freq

    def train(self):
        time = 0
        episode = 1
        loss = None

        self.__env.new_episode()

        while self.__get_phase(time) is not AgentPhase.FINAL:
            if self.__env.is_episode_finished():
                self.new_episode(time)
                episode += 1
                continue
            self.__frame_window.append(self.__current_state_frame())
            initial_state_frames = self.__frame_window.frames()

            action = self.__action_choicer.choice_action(initial_state_frames, self.__epsilon)

            if not self.is_phase(time, AgentPhase.OBSERVATION):
                self.__epsilon.decrement(time)

            rewards = self.__env.make_action(action)

            if self.__env.is_episode_finished():
                self.new_episode(time)
                episode += 1
                continue
            self.__frame_window.append(self.__current_state_frame())
            final_state_frames = self.__frame_window.frames()

            self.__save_state_transition(action, initial_state_frames, final_state_frames, rewards)

            self.__update_target_model_wights(time)

            if self.is_phase(time, AgentPhase.OBSERVATION):
                self.__log_observation(time, episode)
            elif time % self.__time_step_per_train == 0:
                loss = self.__model_train_strategy.train()
                self.__log_exploration(time)

            time += 1

    def new_episode(self, time):
        self.__env.new_episode()
        self.__frame_window.reset()

    def __save_state_transition(self, action, current_frame_window, next_frame_window, rewards):
        state_transition = StateTransition(
            current_frame_window,
            action,
            rewards,
            next_frame_window,
            self.__env.is_episode_finished()
        )
        self.__state_transition_memory.add(state_transition)

    def __current_state_frame(self):
        frame = self.__env.current_state().frame()
        return self.__image_pre_processor.pre_process(frame)

    def __update_target_model_wights(self, time):
        if time % self.__copy_weights_to_target_model_freq == 0:
            self.__model.copy_weights_to(self.__target_model)
            self.__logger.info(f'Update TD Target model weights')

    def __get_phase(self, time):
        if time <= self.__observe_times:
            return AgentPhase.OBSERVATION
        if self.__observe_times < time <= self.__explore_times:
            return AgentPhase.EXPLORATION
        if time >= self.__observe_times + self.__explore_times:
            return AgentPhase.TRAIN

        return AgentPhase.FINAL

    def is_phase(self, time, phase):
        return self.__get_phase(time) is phase

    def __log_observation(self, time, episode):
        phase = self.__get_phase(time)
        self.__logger.info(f'Time:{time}, {phase}, Episode:{episode} - Env:{self.__env.current_state().variables}')

    def __log_exploration(self, time):
        phase = self.__get_phase(time)
        self.__logger.info(f'Time:{time}, {phase}, Ep:{self.__epsilon.value()}')
