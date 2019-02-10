from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentFinalPhase:
    def on_each_time(self, ctx):
        ctx.log('Final', 'End training')
        # ctx.model.save('checkpoints/last_weights.h5')

    def on_episode_finish(self, ctx):
        pass
