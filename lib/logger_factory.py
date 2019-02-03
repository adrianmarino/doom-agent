import logging
import sys
from logging.handlers import TimedRotatingFileHandler


class LoggerFactory():
    def __init__(self, config): self.__config = config

    def create(self):
        logger = logging.getLogger()
        logger.setLevel(self.__config['level'])

        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False

        # Console logger
        logger.addHandler(self.__setup_logger_handler(
            logging.StreamHandler(sys.stdout)
        ))

        # File logger
        logger.addHandler(self.__setup_logger_handler(
            TimedRotatingFileHandler(self.__config['filename'], when='midnight')
        ))

        return logger

    def __setup_logger_handler(self, handler):
        handler.setLevel(
            self.__config['level']
        )
        handler.setFormatter(logging.Formatter(
            fmt=self.__config['message_format'],
            datefmt=self.__config['date_format']
        ))
        return handler
