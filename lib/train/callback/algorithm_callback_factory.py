from lib.train.callback.metric.environment_variable_metric_update_callback import \
    EnvironmentVariableMetricUpdateCallback
from lib.train.callback.metric.epsilon_metric_update_callback import EpsilonMetricUpdateCallback
from lib.train.callback.metric.td_target_metric_update_callback import TDTargetMetricUpdateCallback
from lib.train.callback.model.save_model_callback import SaveModelCallback

class AlgorithmCallbackFactory:
    def __init__(self, cfg):
        self.cfg = cfg

    def create_all(self, names):
        return [self.create(name) for name in names]

    def create(self, name):
        if 'epsilon' == name:
            return EpsilonMetricUpdateCallback(
                self.cfg['callbacks.algorithm.settings.epsilon.metric_path'],
                self.cfg['callbacks.algorithm.settings.epsilon.initial'],
                self.cfg['callbacks.algorithm.settings.epsilon.final'],
                self.cfg['callbacks.algorithm.settings.epsilon.explore']
            )

        if 'td_target_update' == name:
            return TDTargetMetricUpdateCallback(
                self.cfg['callbacks.algorithm.settings.td_target.metric_path'],
                self.cfg['callbacks.algorithm.settings.td_target.schedule']
            )

        if 'kills' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['callbacks.algorithm.settings.kills.metric_path'],
                metric_name='kills',
                display_name='Player Enemy Killed Count',
                description='Enemies killed count at the end of episode.'
            )

        if 'ammo' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['callbacks.algorithm.settings.ammo.metric_path'],
                metric_name='ammo',
                display_name='Episode Final Player Ammo',
                description='Player ammo amount at the end of episode.'
            )

        if 'health' == name:
            return EnvironmentVariableMetricUpdateCallback(
                metric_path=self.cfg['callbacks.algorithm.settings.health.metric_path'],
                metric_name='health',
                display_name='Episode Final Player Health',
                description='Player health level at the end of episode.'
            )

        if 'save_model' == name:
            return SaveModelCallback(
                metric_path=self.cfg['callbacks.algorithm.settings.save_model.path'],
                freq=self.cfg['callbacks.algorithm.settings.save_model.freq']
            )

        raise Exception(f'Not found {name} callback')
