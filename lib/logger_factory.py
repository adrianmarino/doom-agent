import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from lib.util.os_utils import create_directory


class LoggerFactory():
    def __init__(self, config):
        self.__config = config
        self.__file_path = f'{self.__config["path"]}/{self.__config["name"]}.log'
        create_directory(self.__config['path'])
        self.__level = self.__to_logging_level(self.__config['level'])

    def create(self):
        logger = logging.getLogger()
        logger.setLevel(self.__level)

        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False

        # Console logger
        logger.addHandler(self.__setup_logger_handler(
            logging.StreamHandler(sys.stdout)
        ))

        # File logger
        logger.addHandler(self.__setup_logger_handler(
            TimedRotatingFileHandler(filename=self.__file_path, when='midnight')
        ))

        return logger

    @staticmethod
    def __to_logging_level(level):
        return eval(f'logging.{level}')

    def __setup_logger_handler(self, handler):
        handler.setLevel(self.__level)
        handler.setFormatter(logging.Formatter(
            fmt=self.__config['message_format'],
            datefmt=self.__config['date_format']
        ))
        return handler
