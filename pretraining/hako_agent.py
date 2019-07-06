from pommerman.agents import BaseAgent
from pommerman import characters
from agents import hako_agent
import subprocess
from feature_engineer import FeatureEngineer
from pretraining import pretraining_database
import time


class HakoAgent(BaseAgent):

    def __init__(self, id, character=characters.Bomber):
        super(HakoAgent, self).__init__(character)
        self._agent = hako_agent.MyAgent()
        self.feature_engineer = FeatureEngineer()
        self.id = id

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        action = self._agent.act(observation, action_space)
        self.feature_engineer.update_features(observation)
        map, player = self.feature_engineer.get_features()
        pretraining_database.add_data(map, player, action, self.id)
        return action

    def episode_end(self, reward):
        self.feature_engineer = FeatureEngineer()
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()
