from lib.agent.callback.agent_callback import AgentCallback
from lib.metric.tensor_board_metric import TensorBoardMetric


@AgentCallback.register
class TDTargetMetricUpdateCallback:
    def __init__(self, metric_path, update_target_model_freq):
        description = f"""
See time when TD Target model weights were updated 
from main model weights (See: DDQN). Configured with 
train.update_target_model_freq property. 
Updated every {update_target_model_freq} times.
"""

        self.__metric = TensorBoardMetric(
            metric_path=metric_path,
            metric_name='td_target_update',
            display_name='TD Target Model Update',
            description=description
        )
        self.repeat = 0

    def perform(self, ctx):
        self.__metric.update(self.__get_value(ctx), ctx.time)

    def __get_value(self, ctx):
        if ctx.is_td_target_update_time():
            self.repeat = 10

        if self.repeat > 0:
            value = 1
            self.repeat -= 1
        else:
            value = 0

        return value
