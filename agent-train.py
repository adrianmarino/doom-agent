from lib.env.reward.doom_rewards_computation_strategy import DoomRewardsComputationStrategy
from lib.k_session_setup import setup_backend
from lib.params_resolver import ParamsResolver
from lib.train.algorithm.ddqn.ddqn_train_algorithm_factory import DDQNTrainAlgorithmFactory
from lib.train.report.report_utils import build_train_report
from lib.util.config import Config

if __name__ == "__main__":
    setup_backend()
    cfg = Config('./config.yml')
    params = ParamsResolver(cfg, description="Train Doom Agent").resolver()

    # Builder train algorithm...
    rewards_computation_strategy = DoomRewardsComputationStrategy(
        cfg['hiperparams.rewards.kills'],
        cfg['hiperparams.rewards.ammo'],
        cfg['hiperparams.rewards.health']
    )
    train_algorithm = DDQNTrainAlgorithmFactory.create(cfg, rewards_computation_strategy)

    # Train model and generate weights files under checkpoint path...
    train_algorithm.train(params['weights'])

    # Generate report file...
    build_train_report(cfg)
