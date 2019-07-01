from pommerman.agents import BaseAgent
import random
import time
import numpy as np
from FeatureEngineer import FeatureEngineer
import sys
from pommerman import utility


class TestAgent(BaseAgent):
    FeatureEngineer = FeatureEngineer()

    def act(self, observation, action_space):

        # print(sys.getsizeof(observation))
        # self.FeatureEngineer.make_features(observation)
        # time.sleep(0.5)
        # print(observation["teammate"].value)
        # print(observation)
        return 0 #random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)
