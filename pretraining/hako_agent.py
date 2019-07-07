from pommerman.agents import BaseAgent
from pommerman import characters
from agents import hako_agent
from pommerman.runner import DockerAgentRunner
import subprocess
from feature_engineer import FeatureEngineer
from pretraining import pretraining_database
import time


class HakoAgent(BaseAgent):

    def __init__(self, a_id, character=characters.Bomber):
        super(HakoAgent, self).__init__(character)
        self._agent = hako_agent.MyAgent()
        self.feature_engineer = FeatureEngineer()
        self.a_id = a_id

    def act(self, observation, action_space):
        action = self._agent.act(observation, action_space)
        self.feature_engineer.update_features(observation)
        map, player = self.feature_engineer.get_features()
        pretraining_database.add_data(map, player, action, self.a_id)
        return action

    def episode_end(self, reward):
        self.feature_engineer = FeatureEngineer()
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()
