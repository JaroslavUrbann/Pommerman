import numpy as np
from constants import *
from feature_engineer import FeatureEngineer


def create_database(db_size):
    global x, y, feature_engineers, step_index, database_size

    database_size = db_size
    step_index = 0
    x = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")
    y = np.zeros((database_size,), dtype="float32")
    feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


def get_database():
    global x, y
    return x, y


def set_database(x_, y_):
    global x, y
    x = x_
    y = y_


def next_episode():
    global feature_engineers
    feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


def add_data(state, actions):
    global x, y, feature_engineers, step_index, database_size
    for agent in state[0]["alive"]:
        if step_index < database_size:
            i = agent - 10
            features = feature_engineers[i].get_features(state[i])
            x[step_index] = features
            y[step_index] = actions[i]
            step_index += 1
