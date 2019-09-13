from constants import *
from feature_engineer import FeatureEngineer
import numpy as np
import h5py


class DB:
    def __init__(self, size, name="db"):
        self.name = name
        self.size = size
        self.step_index = 0
        self.x = np.zeros((size, 11, 11, N_FEATURES), dtype="float32")
        self.y = np.zeros((size,), dtype="float32")
        self.feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


    def get(self):
        return self.x, self.y


    def set(self, x, y):
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


    def upload(self, drive):
        with h5py.File(self.name, "w") as storage:
            x_db = storage.create_dataset("x", self.x.shape, compression="gzip")
            y_db = storage.create_dataset("y", self.y.shape, compression="gzip")
            x_db[:, :, :, :] = self.x
            y_db[:] = self.y

        database = drive.CreateFile({'title': self.name})
        database.SetContentFile(self.name)
        database.Upload()