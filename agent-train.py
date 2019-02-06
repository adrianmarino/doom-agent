from keras import backend as K

from lib.agent.agent_factory import AgentFactory
from lib.env.environment import Environment
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.util.config import Config
from lib.util.session_builder import SessionBuilder


def setup_session():
    K.set_session(SessionBuilder().regulate_gpu_memory_use().build())

cfg = Config('./config.yml')

if __name__ == "__main__":
    setup_session()

    env = Environment(
        config_file=cfg['env.config_file'],
        advance_steps=cfg['env.train.advance_steps'],
        rewards_computation_strategy=DoomRewardsComputationStrategy(),
        variable_names=cfg['env.variables'],
        window_visible=cfg['env.train.show'],
        sound_enabled=cfg['env.play.sound']
    )
    agent = AgentFactory.create(cfg)

    agent.train()
