from lib.agent.callback.metric.custom_metric_update_callback import CustomMetricUpdateCallback


class EnvironmentVariableMetricUpdateCallback(CustomMetricUpdateCallback):
    def __init__(self, path, variable_name, label):
        super().__init__(
            path=path,
            label=label,
            value_resolver=lambda ctx: ctx.previous_state_variables()[variable_name],
            each_episode=True
        )
