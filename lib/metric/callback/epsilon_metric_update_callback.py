from lib.metric.callback.custom_metric_update_callback import CustomMetricUpdateCallback


class EpsilonMetricUpdateCallback(CustomMetricUpdateCallback):
    def __init__(self, path):
        super().__init__(
            path=path,
            label='Epsilon',
            value_resolver=lambda agent, time, episode: agent.epsilon.value(),
            each_episode=False
        )
