from lib.train.algorithm.ddqn.phase.algorithm_phase import AlgorithmPhase


@AlgorithmPhase.register
class AlgorithmFinalPhase:
    def on_each_time(self, ctx):
        ctx.log('Final', 'End training')
        # ctx.model.save('checkpoints/last_weights.h5')

    def on_episode_finish(self, ctx):
        pass
