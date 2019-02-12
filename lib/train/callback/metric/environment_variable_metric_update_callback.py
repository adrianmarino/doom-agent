from lib.train.callback.metric.custom_metric_update_callback import CustomMetricUpdateCallback


class EnvironmentVariableMetricUpdateCallback(CustomMetricUpdateCallback):
    def __init__(self, metric_path, metric_name, display_name='', description=''):
        super().__init__(
            metric_path=metric_path,
            metric_name=metric_name,
            value_resolver=lambda ctx: ctx.env.previous_state_variables()[metric_name],
            display_name=display_name,
            description=description,
            each_episode=True
        )
