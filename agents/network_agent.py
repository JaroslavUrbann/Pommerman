from pommerman.agents import BaseAgent
from feature_engineer import FeatureEngineer
from pommerman import characters
import numpy as np


class NetworkAgent(BaseAgent):

    def __init__(self, network):
        super(NetworkAgent, self).__init__(characters.Bomber)
        self.feature_engineer = FeatureEngineer()
        self.network = network

    def act(self, observation, action_space):
        features = self.feature_engineer.get_features(observation)
        action, _ = self.network.predict(features)
        return action

    def episode_end(self, reward):
        self.feature_engineer = FeatureEngineer()
