from pommerman.agents import BaseAgent
from feature_engineer import FeatureEngineer
from pommerman import characters
from agents.LN.LNController import *


class LNAgent(BaseAgent):

    def __init__(self, agent_id, network_id, LN, character=characters.Bomber):
        super(LNAgent, self).__init__(character)
        self.feature_engineer = FeatureEngineer()
        self.n_id = network_id
        self.LN_controller = get_controller(self.network_id, LN)
        self.a_id = agent_id
        self._steps = 0
        self.tim = 0

    def act(self, observation, action_space):
        global y1, y2
        tim = time.time()
        self.feature_engineer.update_features(observation)
        map, player = self.feature_engineer.get_features()
        if self.id == 1:
            self.LN_controller.get_prediction_agent1(map, player)
            y = min(y1, 5)
        if self.id == 2:
            self.LN_controller.get_prediction_agent2(map, player)
            y = min(y1, 5)
        self.tim += time.time() - tim
        self._steps += 1
        return int(y)

    def episode_end(self, reward):
        if self.agent_id == 1:
            self.LN_controller.reset_state()
        print("avg decision time: " + str(self.tim / self._steps))
        self.feature_engineer = FeatureEngineer()

    def shutdown(self):
        delete_controller(self.n_id)
