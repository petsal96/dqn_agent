from agent_code.dqn_agent3.dqn import DQNAgent

def setup(self):
    self.Agent = DQNAgent(self.logger)
    self.Agent.Load()
    self.Agent.Train = False

def act(self, game_state: dict):
    state = self.Agent.CalculateFeatures(game_state)
    action = self.Agent.PredictAction(state)
    return action