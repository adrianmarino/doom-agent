from shutil import copyfile

from lib.agent.agent_factory import AgentFactory
from lib.env.environment import Environment
from lib.env.reward.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.k_session_setup import setup_session
from lib.model.model_utils import get_best_weights_file_from, get_loss_model_weights_path
from lib.params_resolver import ParamsResolver
from lib.report.report_factory import AgentReportFactory
from lib.report.report_utils import write_report
from lib.util.config import Config
from lib.util.os_utils import create_file_path
from lib.util.time_utils import str_time


def cp_best_weights_to_reports_path():
    best_weights_file = get_best_weights_file_from(cfg['checkpoint.path'])
    loss = get_loss_model_weights_path(best_weights_file)
    copyfile(
        best_weights_file,
        create_file_path(cfg['report.path'], f'{str_time}-weights-loss_{loss}', 'h5')
    )


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

str_time = str_time()
report = AgentReportFactory.json_report(cfg)

cp_best_weights_to_reports_path()
write_report(cfg['report.path'], report, str_time, 'json')
print(report)
