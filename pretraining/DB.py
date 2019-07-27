import numpy as np
from constants import *
import tensorflow as tf
from feature_engineer import FeatureEngineer

database_size = 0
step_index = 0

x = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")
y = np.zeros((database_size, N_CLASSES), dtype="float32")
feature_engineers = [None, None, None, None]


def create_database(db_size):
    global x, y, feature_engineers, step_index, database_size

    database_size = db_size
    step_index = 0
    x = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")
    y = np.zeros((database_size, N_CLASSES), dtype="float32")
    feature_engineers = [FeatureEngineer(), FeatureEngineer(), FeatureEngineer(), FeatureEngineer()]


def get_database():
    global x, y
    return x, y


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
            y[step_index] = tf.keras.utils.to_categorical(actions[i], num_classes=N_CLASSES)
            step_index += 1
