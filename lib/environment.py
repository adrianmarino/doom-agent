from vizdoom import *


class Environment:
    def __init__(self, config_file, scene_name):
        self.__game = DoomGame()
        self.__game.load_config(config_file)
        self.__game.set_doom_scenario_path(f'{scene_name}.wad')
        self.__game.init()

        self.actions = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

    def new_episode(self): self.__game.new_episode()

    def make_action(self, action): self.__game.make_action(action)

    def is_episode_finished(self): self.__game.is_episode_finished()

    def state(self): return self.__game.get_state().screen_buffer
