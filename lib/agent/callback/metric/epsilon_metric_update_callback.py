from lib.agent.callback.metric.custom_metric_update_callback import CustomMetricUpdateCallback


class EpsilonMetricUpdateCallback(CustomMetricUpdateCallback):
    def __init__(self, metric_path, initial, final, explore_times):
        description = f"""
Used to get action from model using epsilon-greedy policy. This value is 
between 1 and a configured min value. On the other hand, is decremented 
with delta=(initial-final)/explore_times in each time increment under 
explore phase. Handled with epsilon.initial, epsilon.final and 
phase_time.explore properties.

    Parameters:
        - Initial: {initial}
        - Final: {final}
        - explore_times  {explore_times}
        - Delta time: -{(initial - final) / explore_times:.10f}
"""

        super().__init__(
            metric_path=metric_path,
            metric_name='epsilon',
            value_resolver=lambda ctx: ctx.epsilon.value(),
            display_name='Epsilon-Greedy Value',
            description=description,
            each_episode=False
        )
