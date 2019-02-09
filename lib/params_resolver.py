import argparse

from lib.model.model_utils import get_best_weights_file_from


class ParamsResolver:
    def __init__(self, cfg, description):
        self.__parser = argparse.ArgumentParser(description=description)

        default_weights_path = get_best_weights_file_from(cfg['checkpoint.path'])
        self.__parser.add_argument(
            '--weights',
            help=f'Model weights file. As default get best loss weights file if exists. Default value: {default_weights_path}',
            default=default_weights_path
        )

        episodes = cfg['env.play.episodes']
        self.__parser.add_argument(
            '--episodes',
            help=f'Number of episodes to play. Default value: {episodes}.',
            default=episodes
        )

        frame_delay = cfg['env.play.frame_delay']
        self.__parser.add_argument(
            '--frame-delay',
            help=f'Delay time between frames when play agent. Default value: {frame_delay}.',
            default=frame_delay
        )

    def resolver(self):
        return self.__params(self.__parser)

    @staticmethod
    def __params(parser):
        return {k: v for k, v in dict(parser.parse_args()._get_kwargs()).items()}
