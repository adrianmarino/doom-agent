import argparse

from lib.util.file_utils import last_created_file_from


class ParamsResolver:
    def __init__(self, cfg, description):
        self.__default_weights_path = last_created_file_from(cfg['checkpoint.path'] + '/*.h5')
        self.__play_episodes = cfg['env.play.episodes']
        self.__frame_delay = cfg['env.play.frame_delay']
        self.__parser = argparse.ArgumentParser(description=description)

    def resolver(self):
        self.__parser.add_argument(
            '--weights',
            help='model weights file.',
            default=self.__default_weights_path
        )

        self.__parser.add_argument(
            '--episodes',
            help='Number of episodes to play.',
            default=self.__play_episodes
        )

        self.__parser.add_argument(
            '--frame-delay',
            help='Delay time between frames',
            default=self.__frame_delay
        )

        return self.__params(self.__parser)

    @staticmethod
    def __params(parser):
        return {k: v for k, v in dict(parser.parse_args()._get_kwargs()).items()}
