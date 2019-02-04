class DefaultRewardsComputationStrategy:
    def calculate(self, env, initial_state, final_state): return env.accumulated_rewards()
