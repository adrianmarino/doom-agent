from lib.agent.frame_window import FrameWindow
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
            action_choicer,
            state_transition_memory,
            image_pre_processor,
            logger,
            observe_times=5000,
            explore_times=50000,
            save_model_freq=1000,
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
        self.logger = logger

        self.__observe_times = observe_times
        self.__explore_times = explore_times
        self.__save_model_freq = save_model_freq
        self.__time_step_per_train = time_step_per_train
        self.__copy_weights_to_target_model_freq = copy_weights_to_target_model_freq


    def train(self, full_train_times=0):
        time = 0
        self.__env.new_episode()
        while self.__train_while(time, full_train_times):
            initial_state_frames = self.__frame_window.append(self.__current_state_frame()).frames
            if self.__env.is_episode_finished():
                self.new_episode(time)
                continue

            action = self.__choice_action(time, initial_state_frames)
            rewards = self.__env.make_action(action)

            final_state_frames = self.__frame_window.append(self.__current_state_frame()).frames
            if self.__env.is_episode_finished():
                self.new_episode(time)
                continue

            self.__save_state_transition(action, initial_state_frames, final_state_frames, rewards)

            self.logger.info(f'Time: {time} - Board: {self.__env.current_state().variables}')

            self.__update_target_model_wights(time)
            self.__train_model(time)
            self.__save_model(time)
            time += 1

    def new_episode(self, time):
        print(f'Time: {time} - Finish Episode!')
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

    def __train_while(self, time, full_train_times):
        return time <= (self.__observe_times + self.__explore_times + full_train_times) and \
               not self.__env.is_episode_finished()

    def __save_model(self, time):
        if time % self.__save_model_freq == 0:
            self.__model.save('models/model.h5')

    def __train_model(self, time):
        if time > self.__observe_times and time % self.__time_step_per_train == 0:
            max_q_value, loss = self.__model_train_strategy.train()
            print('loss: ', loss, ', Max Q value: ', max_q_value)

    def __update_target_model_wights(self, time):
        if time % self.__copy_weights_to_target_model_freq == 0:
            self.__model.copy_weights_to(self.__target_model)

    def __choice_action(self, time, current_frame_window):
        action = self.__action_choicer.choice_action(current_frame_window, self.__epsilon)
        self.__epsilon.decrement(time)
        return action
