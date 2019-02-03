from vizdoom import *

from lib.rewards.default_rewards_calculation_strategy import DefaultRewardsComputationStrategy
from lib.state.environment_state import EnvironmentState
from lib.util.on_hot_vector_factory import OnHotVectorFactory

class Environment:
    def __init__(
            self,
            config_file,
            advance_steps,
            variable_names,
            rewards_computation_strategy=DefaultRewardsComputationStrategy(),
            sound_enabled=False,
            screen_resolution=ScreenResolution.RES_640X480,
            window_visible=True
    ):
        self.__game = DoomGame()
        self.__game.load_config(config_file)
        self.__game.set_sound_enabled(sound_enabled)
        self.__game.set_screen_resolution(screen_resolution)
        self.__game.set_window_visible(window_visible)
        self.__game.init()
        self.advance_steps = advance_steps
        self.rewards_computation_strategy = rewards_computation_strategy
        self.__variable_names = variable_names


    def new_episode(self): self.__game.new_episode()

    def make_action(self, action):
        initial_state = self.current_state()
        if(self.is_episode_finished()):
            return 0

        self.__make_action(action)
        self.__advance_states(self.advance_steps)

        final_state = self.current_state()
        if(self.is_episode_finished()):
            return 0

        return self.rewards_computation_strategy.calculate(self, initial_state, final_state)

    def accumulated_rewards(self): return self.__game.get_last_reward()

    def is_episode_finished(self): return self.__game.is_episode_finished()

    def __make_action(self, action_number):
        action = OnHotVectorFactory.create(size=self.actions_count(), one_index=action_number)
        self.__game.make_action(action)

    def __advance_states(self, count): self.__game.advance_action(count)

    def current_state(self):
        return EnvironmentState(self.__game.get_state(), self.__variable_names)

    def actions_count(self): return self.__game.get_available_buttons_size()
