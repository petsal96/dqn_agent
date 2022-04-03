from agent_code.dqn_agent3.dqn import ACTIONS, MIN_MEMORY_BUFFER, DQNAgent
import events as e
import sys
import matplotlib.pyplot as plt
import numpy as np

MAX_EVAL_GAMES = 50
MAX_GAMES_FOR_EVAL = 500
MAX_STEPS_FOR_TARGET_UPDATE = 1000

def setup_training(self):

    self.NoOfSteps = 0
    
    self.NoOfEpisodesPlotX = []
    self.AverageLossY = []
    self.AverageScoreY = []
    
    self.Score = 0
    self.Episode = 0
    self.EvalEpisode = 0
    self.Eval = False

    self.Agent.Train = True
    self.Agent.logger = self.logger

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    if old_game_state is None:
        return

    reward = self.Agent.CalculateRewards(events)

    if self.Eval:
        self.Score += reward
    else:
        self.NoOfSteps += 1
        self.Agent.StoreExperience(old_game_state, self_action, new_game_state, reward, False)
        self.Agent.UpdateWeights()

        self.Agent.UpdateExplorationProbability()

def end_of_round(self, last_game_state: dict, last_action: str, events: list):

    self.Episode += 1

    reward = self.Agent.CalculateRewards(events)

    if self.Eval:
        self.Score += reward
        self.EvalEpisode += 1
        if self.EvalEpisode >= MAX_EVAL_GAMES:
            self.NoOfEpisodesPlotX.append(self.Episode)
            self.AverageScoreY.append(self.Score/MAX_EVAL_GAMES)
            self.EvalEpisode = 0
            self.Eval = False
            self.Score = 0
            self.Agent.Train = True
            plt.plot(self.NoOfEpisodesPlotX, self.AverageScoreY)
            plt.savefig("avg_score.png")
            plt.close()
    else:
        self.NoOfSteps += 1
        self.Agent.StoreExperience(last_game_state, last_action, last_game_state, reward, False)
        self.Agent.UpdateWeights()

        self.Agent.UpdateExplorationProbability()

    if not self.Eval and self.NoOfSteps > MAX_STEPS_FOR_TARGET_UPDATE and len(self.Agent.MemoryBuffer) >= MIN_MEMORY_BUFFER:
        self.Agent.UpdateTargetModel()
        self.Agent.Save()
        self.NoOfSteps = 0

    if self.Episode % MAX_GAMES_FOR_EVAL == 0:
        self.Eval = True
        self.Agent.Train = False
