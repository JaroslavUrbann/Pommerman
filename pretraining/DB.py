from constants import *
from feature_engineer import FeatureEngineer
import numpy as np


class DB:
    def __init__(self, db_size):
        self.size = db_size
        self.step_index = 0
        self.x = np.zeros((db_size, 11, 11, N_FEATURES), dtype="float32")
        self.y = np.zeros((db_size,), dtype="float32")
        self.feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


    def get_database(self):
        return self.x, self.y


    def set_database(self, x, y):
        self.x = x
        self.y = y


    def next_episode(self):
        self.feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


    def add_data(self, state, actions):
        for agent in state[0]["alive"]:
            if self.step_index < self.size:
                i = agent - 10
                features = self.feature_engineers[i].get_features(state[i])
                self.x[self.step_index] = features
                self.y[self.step_index] = actions[i]
                self.step_index += 1
