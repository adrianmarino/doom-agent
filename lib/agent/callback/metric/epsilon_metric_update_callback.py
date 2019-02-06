from lib.agent.callback.metric.custom_metric_update_callback import CustomMetricUpdateCallback

class EpsilonMetricUpdateCallback(CustomMetricUpdateCallback):
    def __init__(self, metric_path):
        super().__init__(
            metric_path=metric_path,
            metric_name='epsilon',
            value_resolver=lambda ctx: ctx.epsilon.value(),
            display_name='Epsilon-Greedy Value',
            description=f'Used to get action from model using epsilon-greedy policy. This value is between 1 and a configured min value. On the other hand, is decremented with delta=(initial-final)/explore_times in each time increment under explore phase. Handled with epsilon.initial, epsilon.final and phase_time.explore properties.',
            each_episode=False
        )
