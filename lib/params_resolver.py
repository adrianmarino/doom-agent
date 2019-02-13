import argparse


class ParamsResolver:
    def __init__(self, description):
        self.__parser = argparse.ArgumentParser(description)

        self.__parser.add_argument(
            '--weights',
            help=f'Model weights file. As default get best loss weights file if exists.'
        )
        self.__parser.add_argument('--config', help=f'Agent train and & play settings.')

    def resolver(self):
        return self.__params(self.__parser)

    @staticmethod
    def __params(parser):
        return {k: v for k, v in dict(parser.parse_args()._get_kwargs()).items()}
