from lib.train.algorithm.ddqn.phase.algorithm_phase import AlgorithmPhase


@AlgorithmPhase.register
class AlgorithmFinalPhase:
    def on_each_time(self, ctx):
        ctx.log('Final', 'End training')

    def on_episode_finish(self, ctx):
        pass
