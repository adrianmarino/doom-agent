class AgentContext:
    def __init__(
            self,
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
    ):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.model_train_strategy = model_train_strategy
        self.epsilon = epsilon
        self.logger = logger
        self.train_freq = train_freq
        self.td_target_update_freq_resolver = td_target_update_freq_resolver
        self.observe_times = observe_times
        self.explore_times = explore_times
        self.train_times = train_times
        self.reset()

    def reset(self):
        self.time = 0
        self.episode = 0

    def previous_state_variables(self):
        return self.env.previous_state().variables

    def update_epsilon(self):
        self.epsilon.decrement()

    def is_train_time(self):
        return not self.is_final_time() and self.time % self.train_freq == 0

    def is_td_target_update_time(self):
        return self.time % self.td_target_update_freq_resolver.resolve(self.time) == 0

    def train_model(self):
        self.model_train_strategy.train()

    def update_td_target_model(self):
        self.model.copy_weights_to(self.target_model)

    def epsilon_value(self):
        return self.epsilon.value()

    def increase_time(self):
        self.time += 1
        return self.time

    def increase_episode(self):
        self.episode += 1
        return self.episode

    def is_episode_finished(self):
        return self.env.is_episode_finished()

    def log(self, phase, message):
        self.logger.info(f'Episode:{self.episode} - Time:{self.time} - Phase:{phase} - {message}')

    def is_final_time(self):
        return self.time == (self.observe_times + self.explore_times + self.train_times)
