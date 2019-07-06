from pommerman.agents import BaseAgent
import random
import time
import numpy as np
from feature_engineer import FeatureEngineer
from pommerman import characters
from constants import *

LN = None
x1_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
x2_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
x1_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
x2_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
y1, y2 = 0, 0
agent2_ready = False
is_agent1_dead = False
is_agent2_dead = False
prediction_done = False


def reset_state():
    global x1_map, x2_map, x1_player, x2_player, step_index, agent1_ready, agent2_ready, is_agent1_dead, is_agent2_dead, prediction_done, y1, y2
    x1_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
    x2_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
    x1_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    x2_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    y1, y2 = 0, 0
    agent2_ready = False
    is_agent1_dead = False
    is_agent2_dead = False
    prediction_done = False


def get_prediction_agent1(map, player):
    global x1_map, x2_map, x1_player, x2_player, step_index, agent1_ready, agent2_ready, is_agent1_dead, is_agent2_dead, prediction_done, y1, y2, LN
    x1_map[0] = map
    x1_player[0] = player
    tim = time.time()

    while True:
        if agent2_ready or is_agent2_dead or time.time() - tim > 0.3:
            if time.time() - tim >= 0.3:
                is_agent2_dead = True
            if is_agent2_dead:
                out1, out2 = LN.predict(x1_map, np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32"), x2_player, np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32"))
            if agent2_ready:
                out1, out2 = LN.predict(x1_map, x2_map, x2_player, x2_player)
            y1 = np.argmax(out1)
            y2 = np.argmax(out2)
            prediction_done = True
            return


def get_prediction_agent2(map, player):
    global x1_map, x2_map, x1_player, x2_player, step_index, agent1_ready, agent2_ready, is_agent1_dead, is_agent2_dead, prediction_done, y1, y2, LN
    x2_map[0] = map
    x2_player[0] = player
    agent2_ready = True
    tim = time.time()

    while True:
        if prediction_done or is_agent1_dead or time.time() - tim > 0.3:
            if time.time() - tim >= 0.3:
                is_agent1_dead = True
            if is_agent1_dead:
                _, out2 = LN.predict(np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32"), x2_map, np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32"), x2_player)
                y2 = np.argmax(out2)
            prediction_done = False
            agent2_ready = False
            return


class TestingAgent(BaseAgent):

    def __init__(self, id, character=characters.Bomber):
        super(TestingAgent, self).__init__(character)
        self.feature_engineer = FeatureEngineer()
        self.id = id
        self._steps = 0
        self.tim = 0

    def act(self, observation, action_space):
        global y1, y2
        tim = time.time()
        self.feature_engineer.update_features(observation)
        map, player = self.feature_engineer.get_features()
        if self.id == 1:
            get_prediction_agent1(map, player)
            y = min(y1, 5)
        if self.id == 2:
            get_prediction_agent2(map, player)
            y = min(y1, 5)
        self.tim += time.time() - tim
        self._steps += 1
        return int(y)

    def episode_end(self, reward):
        if self.agent_id == 1:
            reset_state()
        print("avg decision time: " + str(self.tim / self._steps))
        self.feature_engineer = FeatureEngineer()
