from lib.train.algorithm.ddqn.phase.algorithm_phase import AlgorithmPhase


@AlgorithmPhase.register
class AlgorithmExplorePhase:
    def on_each_time(self, ctx):
        ctx.epsilon.decrement()
        if ctx.is_train_time():
            ctx.model_train_strategy.train()

        if ctx.is_td_target_update_time():
            ctx.model.copy_weights_to(ctx.target_model)
            ctx.log('Explore', f'Update TD Target Model (Each {ctx.td_target_update_freq_resolver.current()} times)')

    def on_episode_finish(self, ctx):
        ctx.log('Explore', f'Env:{ctx.env.previous_state_variables()} - Epsilon:{ctx.epsilon}')
