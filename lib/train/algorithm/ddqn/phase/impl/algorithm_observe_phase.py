from lib.train.algorithm.ddqn.phase.algorithm_phase import AlgorithmPhase


@AlgorithmPhase.register
class AlgorithmObservePhase:
    def on_each_time(self, ctx):
        pass

    def on_episode_finish(self, ctx):
        ctx.log('Observe', f'Env:{ctx.env.previous_state_variables()}')
