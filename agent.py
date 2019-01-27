from lib.frame_window import FrameWindow


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
            observe_times = 5000,
            explore_times = 50000,
            save_model_freq = 1000,
            copy_weights_to_target_model_freq = 3000
    ):
        self.__env = env
        self.__model = model
        self.__target_model = target_model
        self.__model_train_strategy = model_train_strategy
        self.__epsilon = epsilon
        self.__action_choicer = action_choicer
        self.__state_transition_memory = state_transition_memory
        self.__frame_window = FrameWindow(frame_shape=(input_shape.rows, input_shape.cols), size=input_shape.channels)

        self.__observe_times = observe_times
        self.__explore_times = explore_times
        self.__save_model_freq = save_model_freq
        self.__copy_weights_to_target_model_freq = copy_weights_to_target_model_freq

    def train(self, train_times=0):
        time = 0
        self.__env.new_episode()

        while self.__train_while(time, train_times):
            if self.__env.is_episode_finished(): self.__env.new_episode()

            self.__frame_window.append(self.__env.current_state())

            action = self.__action_choicer.choice_action(self.__env.current_state(), self.__epsilon)
            self.__epsilon.decrement(time)

            state_transition = self.__env.make_action(action)
            self.__state_transition_memory.add(state_transition)

            self.__update_target_model_wights(time)
            self.__train_model(time)
            self.__save_model(time)

    def play(self):
        pass

    def __train_while(self, time, train_times):
        return time >= (self.__observe_times + self.__explore_times + train_times) and self.__env.is_episode_finished()

    def __save_model(self, time):
        if time % self.__save_model_freq == 0: self.__model.save("models/model.h5")

    def __train_model(self, time):
        if time < self.__observe_times and time < self.__observe_times + self.__explore_times:
            max_q_value, loss = self.__model_train_strategy.train()
            print('loss: ', loss, ', Max Q value: ', max_q_value)

    def __update_target_model_wights(self, time):
        if time % self.__copy_weights_to_target_model_freq == 0: self.__model.copy_weights_to(self.__target_model)
