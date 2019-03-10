from lib.env.environment import Environment
from lib.logger_factory import LoggerFactory
from lib.player.agent_player import AgentPlayer
from lib.train.model.frame_window_to_model_input_converter import FrameWindowToModelInputConverter
from lib.train.model.image_pre_processor import ImagePreProcessor
from lib.train.model.model_factory import ModelFactory
from lib.util.input_shape import InputShape


class AgentPlayerFactory:
    def __init__(self, logger):
        self.__logger = logger

    def create(self, cfg):
        env = Environment(
            config_file=cfg['env.config_file'],
            advance_steps=cfg['env.play.advance_steps'],
            variable_names=cfg['env.variables'],
            window_visible=cfg['env.play.show'],
            sound_enabled=cfg['env.play.sound']
        )

        input_shape = InputShape.from_str(cfg['hiperparams.input_shape'])

        input_converter = FrameWindowToModelInputConverter()

        model_factory = ModelFactory(input_converter, self.__logger)

        model = model_factory.create(
            cfg['hiperparams.model'],
            input_shape,
            env.actions_count(),
            cfg['hiperparams.lr']
        )

        image_pre_processor = ImagePreProcessor(
            input_shape.rows,
            input_shape.cols,
            cfg['hiperparams.chop_bottom_height']
        )

        return AgentPlayer(
            env,
            input_shape,
            model,
            image_pre_processor,
            self.__logger,
            cfg['env.play.frame_delay'],
            cfg['env.play.episodes']
        )
