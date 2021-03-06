from lib.env.reward.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.k_session_setup import setup_backend
from lib.logger_factory import LoggerFactory
from lib.params_resolver import ParamsResolver
from lib.train.algorithm.ddqn.ddqn_train_algorithm_factory import DDQNTrainAlgorithmFactory
from lib.train.report.formatter.json_report_formatter import PrettyJsonFormatter
from lib.train.report.report_utils import build_train_report
from lib.util.config import Config


def setup():
    setup_backend()
    params = ParamsResolver("Train Agent").resolver()
    cfg = Config(params['config'])
    logger = LoggerFactory(cfg['logger']).create()
    return params, cfg, logger


if __name__ == "__main__":
    params, cfg, logger = setup()

    logger.info(f'Config: {PrettyJsonFormatter().format(cfg.dict)}')

    # Builder train algorithm...
    rewards_computation_strategy = DoomRewardsComputationStrategy(logger, cfg['hiperparams.rewards'])

    train_algorithm = DDQNTrainAlgorithmFactory(logger).create(cfg, rewards_computation_strategy)

    # Train model and generate weights files under checkpoint path...
    train_algorithm.train(params['weights'])

    # Generate report file...
    build_train_report(cfg)
