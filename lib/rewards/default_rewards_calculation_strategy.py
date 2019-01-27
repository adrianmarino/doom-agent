class DefaultRewardsComputationStrategy:
    def calculate(self, env, current_state, next_state): return env.accumulated_rewards()
