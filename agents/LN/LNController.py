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

    def get_prediction_agent1(self, map, player):
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

    def get_prediction_agent2(self, map, player):
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

