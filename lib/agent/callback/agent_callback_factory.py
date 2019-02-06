from lib.agent.callback.metric.environment_variable_metric_update_callback import \
    EnvironmentVariableMetricUpdateCallback
from lib.agent.callback.metric.epsilon_metric_update_callback import EpsilonMetricUpdateCallback
from lib.agent.callback.metric.td_target_metric_update_callback import TDTargetMetricUpdateCallback
from lib.agent.callback.model.save_model_callback import SaveModelCallback


class AgentCallbackFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def create_all(self, names):
        return [self.create(name) for name in names]

    def create(self, name):
        if 'epsilon' == name:
            return EpsilonMetricUpdateCallback(
                self.cfg['metric.path'],
                self.cfg['epsilon.initial'],
                self.cfg['epsilon.final'],
                self.cfg['phase_time.explore']
            )

        if 'td_target_update' == name:
            return TDTargetMetricUpdateCallback(
                self.cfg['metric.path'],
                self.cfg['train.update_target_model_freq']
            )

        if 'kills' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['metric.path'],
                metric_name='kills',
                display_name='Player Enemy Killed Count',
                description='Enemies killed count at the end of episode.'
            )

        if 'ammo' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['metric.path'],
                metric_name='ammo',
                display_name='Episode Final Player Ammo',
                description='Player ammo amount at the end of episode.'
            )

        if 'health' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['metric.path'],
                metric_name='health',
                display_name='Episode Final Player Health',
                description='Player health level at the end of episode.'
            )

        if 'save_model' == name:
            return SaveModelCallback(
                metric_path=self.cfg['train.checkpoint.path'],
                freq=self.cfg['train.checkpoint.freq']
            )

        raise Exception(f'Not found {name} callback')
