from pommerman.agents import BaseAgent
from feature_engineer import FeatureEngineer
from pommerman import characters
from agents.LN.LNController import *


class LNAgent(BaseAgent):

    def __init__(self, a_id, n_id, LN):
        super(LNAgent, self).__init__(characters.Bomber)
        self.feature_engineer = FeatureEngineer()
        self.n_id = a_id
        self.LN_controller = get_controller(self.n_id, LN)
        self.a_id = n_id
        self._steps = 0
        self.tim = 0

    def act(self, observation, action_space):
        tim = time.time()
        self.feature_engineer.update_features(observation)
        map, player = self.feature_engineer.get_features()
        if self.a_id == 1:
            y = self.LN_controller.get_prediction_agent1(map, player)
        if self.a_id == 2:
            y = self.LN_controller.get_prediction_agent2(map, player)
        self.tim += time.time() - tim
        self._steps += 1
        return int(min(y, 5))

    def episode_end(self, reward):
        if self.agent_id == 1:
            self.LN_controller.reset_state()
        self.feature_engineer = FeatureEngineer()

    def shutdown(self):
        print("avg decision time: " + str(self.tim / self._steps))
        delete_controller(self.n_id)
