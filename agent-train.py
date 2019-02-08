from lib.env.environment import Environment

from lib.agent.agent_factory import AgentFactory
from lib.k_session_setup import setup_session
from lib.params_resolver import ParamsResolver
from lib.rewards.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.util.config import Config

cfg = Config('./config.yml')
params = ParamsResolver(cfg, description="Train Doom Agent").resolver()
setup_session()

rewards_computation_strategy = DoomRewardsComputationStrategy(
    cfg['hiperparams.rewards.kills'],
    cfg['hiperparams.rewards.ammo'],
    cfg['hiperparams.rewards.health']
)
env = Environment(
    config_file=cfg['env.config_file'],
    advance_steps=cfg['env.train.advance_steps'],
    rewards_computation_strategy=rewards_computation_strategy,
    variable_names=cfg['env.variables'],
    window_visible=cfg['env.train.show'],
    sound_enabled=cfg['env.train.sound']
)
AgentFactory \
    .create(cfg, env) \
    .train(weights_path=params['weights'])
