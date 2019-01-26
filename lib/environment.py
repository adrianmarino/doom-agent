from vizdoom import *
from lib.on_hot_vector_factory import OnHotVectorFactory


class CheckPoint:
    def __init__(self, current_state, action, rewards, next_state):
        self.current_state = current_state
        self.action = action
        self.rewards = rewards
        self.next_state = next_state

class Environment:
    def __init__(
            self,
            config_file,
            advance_steps,
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

    def new_episode(self): self.__game.new_episode()

    def make_action(self, action):
        current_state = self.current_state()
        self.__make_action(action)
        self.__advance_states(self.advance_steps)

        return CheckPoint(
            current_state=current_state,
            action=action,
            rewards=self.accumulated_rewards(),
            next_state=self.current_state()
        )

    def accumulated_rewards(self): return self.__game.get_last_reward()

    def is_episode_finished(self): self.__game.is_episode_finished()

    def current_state(self): return self.__game.get_state().screen_buffer

    def possible_actions(self): return self.__game.game_variables

    def possible_actions_size(self): return len(self.possible_actions())

    def __make_action(self, action_number):
        action = OnHotVectorFactory.create(size=self.possible_actions_size(), one_index=action_number)
        self.__game.make_action(action)

    def __advance_states(self, count): self.__game.advance_action(count)
