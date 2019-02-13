from lib.k_session_setup import setup_backend
from lib.logger_factory import LoggerFactory
from lib.params_resolver import ParamsResolver
from lib.player.agent_player_factory import AgentPlayerFactory
from lib.util.config import Config


def setup():
    setup_backend()
    params = ParamsResolver("Play Doom Agent").resolver()
    cfg = Config(params['config'])
    logger = LoggerFactory(cfg['logger']).create()
    return params, cfg, logger


if __name__ == "__main__":
    params, cfg, logger = setup()

    # Play agent...
    AgentPlayerFactory(logger).create(cfg).play(params['weights'])
