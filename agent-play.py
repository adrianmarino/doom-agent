from lib.k_session_setup import setup_backend
from lib.params_resolver import ParamsResolver
from lib.player.agent_player_factory import AgentPlayerFactory
from lib.util.config import Config

if __name__ == "__main__":
    params = ParamsResolver("Play Doom Agent").resolver()
    cfg = Config(params['config'])

    setup_backend()

    # Play agent...
    AgentPlayerFactory.create(cfg).play(params['weights'])
