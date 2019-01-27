from vizdoom import *

from lib.state_transition import StateTransition
from lib.rewards.default_rewards_calculation_strategy import DefaultRewardsComputationStrategy
from lib.on_hot_vector_factory import OnHotVectorFactory


class EnvironmentState:
    def __init__(self, game_state): self.__game_state = game_state

    def frame(self): return self.__get_state.screen_buffer

    def variables(self): return self.__get_state.game_variables


class Environment:
    def __init__(
            self,
            config_file,
            advance_steps,
            rewards_computation_strategy=DefaultRewardsComputationStrategy(),
            sound_enabled=False,
            screen_resolution=ScreenResolution.RES_640X480,
            window_visible=False
    ):
        self.__game = DoomGame()
        self.__game.load_config(config_file)
        self.__game.set_sound_enabled(sound_enabled)
        self.__game.set_screen_resolution(screen_resolution)
        self.__game.set_window_visible(window_visible)
        self.__game.init()
        self.advance_steps = advance_steps
        self.rewards_computation_strategy = rewards_computation_strategy

    def new_episode(self): self.__game.new_episode()

    def make_action(self, action):
        current_state = self.current_state()

        self.__make_action(action)
        self.__advance_states(self.advance_steps)

        next_state = self.current_state()
        rewards = self.rewards_computation_strategy.calculate(self, current_state, next_state)
        episode_finished = self.is_episode_finished()

        return StateTransition(current_state, action, rewards, next_state, episode_finished)

    def accumulated_rewards(self): return self.__game.get_last_reward()

    def is_episode_finished(self): return self.__game.is_episode_finished()

    def possible_actions(self): return self.__game.game_variables

    def possible_actions_size(self): return len(self.possible_actions())

    def __make_action(self, action_number):
        action = OnHotVectorFactory.create(size=self.possible_actions_size(), one_index=action_number)
        self.__game.make_action(action)

    def __advance_states(self, count): self.__game.advance_action(count)

    def current_state(self): return EnvironmentState(self.__game.get_state())


