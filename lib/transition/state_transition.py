class StateTransition:
    def __init__(self, start_state, action, rewards, end_state, final):
        self.start_state = start_state
        self.action = action
        self.rewards = rewards
        self.end_state = end_state
        self.final = final
