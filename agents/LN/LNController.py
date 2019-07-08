from constants import *
import time
import numpy as np


class LNController:
    x1_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
    x2_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
    x1_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    x2_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    y1, y2 = 0, 0
    agent2_ready = False
    is_agent1_dead = False
    is_agent2_dead = False
    prediction_done = False

    def __init__(self, LN):
        self.LN = LN

    def reset_state(self):
        self.x1_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
        self.x2_map = np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32")
        self.x1_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
        self.x2_player = np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32")
        self.y1, self.y2 = 0, 0
        self.agent2_ready = False
        self.is_agent1_dead = False
        self.is_agent2_dead = False
        self.prediction_done = False

    def get_prediction_agent1(self, map, player):
        self.x1_map[0] = map
        self.x1_player[0] = player
        tim = time.time()

        while True:
            if self.agent2_ready or self.is_agent2_dead or time.time() - tim > 0.3:
                if time.time() - tim >= 0.3:
                    self.is_agent2_dead = True
                if self.is_agent2_dead:
                    out1, out2 = self.LN.predict(self.x1_map, np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32"), self.x2_player, np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32"))
                if self.agent2_ready:
                    out1, out2 = self.LN.predict(self.x1_map, self.x2_map, self.x2_player, self.x2_player)
                self.y1 = np.argmax(out1)
                self.y2 = np.argmax(out2)
                self.prediction_done = True
                return self.y1

    def get_prediction_agent2(self, map, player):
        self.x2_map[0] = map
        self.x2_player[0] = player
        self.agent2_ready = True
        tim = time.time()

        while True:
            if self.prediction_done or self.is_agent1_dead or time.time() - tim > 0.3:
                if time.time() - tim >= 0.3:
                    self.is_agent1_dead = True
                if self.is_agent1_dead:
                    _, out2 = self.LN.predict(np.zeros((1, 11, 11, N_MAP_FEATURES), dtype="float32"), self.x2_map, np.zeros((1, 11, 11, N_PLAYER_FEATURES), dtype="float32"), self.x2_player)
                    self.y2 = np.argmax(out2)
                self.prediction_done = False
                self.agent2_ready = False
                return self.y2


LNC1 = None
LNC2 = None


def get_controller(network_id, LN):
    global LNC1, LNC2
    if network_id == 1:
        if not LNC1:
            LNC1 = LNController(LN)
        return LNC1
    if network_id == 2:
        if not LNC2:
            LNC2 = LNController(LN)
        return LNC2


def delete_controller(network_id):
    global LNC1, LNC2
    if network_id == 1:
        LNC1 = None
    if network_id == 2:
        LNC2 = None

