from keras import backend as K

from lib.agent.agent_factory import AgentFactory
from lib.env.environment import Environment
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.util.config import Config
from lib.util.file_utils import last_created_file_from
from lib.util.session_builder import SessionBuilder


def setup_session():
    K.set_session(SessionBuilder().regulate_gpu_memory_use().build())


cfg = Config('./config.yml')

if __name__ == "__main__":
    setup_session()

    rewards_computation_strategy = DoomRewardsComputationStrategy(
        cfg['hiperparams.rewards.kills'],
        cfg['hiperparams.rewards.ammo'],
        cfg['hiperparams.rewards.health']
    )

    env = Environment(
        config_file=cfg['env.config_file'],
        advance_steps=cfg['env.play.advance_steps'],
        rewards_computation_strategy=rewards_computation_strategy,
        variable_names=cfg['env.variables'],
        window_visible=cfg['env.play.show'],
        sound_enabled=cfg['env.play.sound']
    )
    agent = AgentFactory.create(cfg, env)

    agent.play(
        episodes=3,
        frame_delay=1/8,
        weights_path = last_created_file_from(cfg['checkpoint.path'] + '/*.h5')
    )
