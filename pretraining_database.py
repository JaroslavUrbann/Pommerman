import numpy as np
from constants import *
import tensorflow as tf

step_index = 0
x1_map = np.zeros((DATABASE_SIZE, 11, 11, N_MAP_FEATURES), dtype="float32")
x2_map = np.zeros((DATABASE_SIZE, 11, 11, N_MAP_FEATURES), dtype="float32")
x1_player = np.zeros((DATABASE_SIZE, 11, 11, N_PLAYER_FEATURES), dtype="float32")
x2_player = np.zeros((DATABASE_SIZE, 11, 11, N_PLAYER_FEATURES), dtype="float32")

_default_y = np.zeros(N_CLASSES, dtype="float32")
_default_y[N_CLASSES - 1] = 1

y1 = np.tile(_default_y, (DATABASE_SIZE, 1))
y2 = np.tile(_default_y, (DATABASE_SIZE, 1))


def reset_database():
    global x1_map, x2_map, x1_player, x2_player, y1, y2, step_index
    step_index = 0
    x1_map = np.zeros((DATABASE_SIZE, 11, 11, N_MAP_FEATURES), dtype="float32")
    x2_map = np.zeros((DATABASE_SIZE, 11, 11, N_MAP_FEATURES), dtype="float32")
    x1_player = np.zeros((DATABASE_SIZE, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    x2_player = np.zeros((DATABASE_SIZE, 11, 11, N_PLAYER_FEATURES), dtype="float32")

    _default_y = np.zeros(N_CLASSES, dtype="float32")
    _default_y[N_CLASSES - 1] = 1

    y1 = np.tile(_default_y, (DATABASE_SIZE, 1))
    y2 = np.tile(_default_y, (DATABASE_SIZE, 1))


def add_data(map, player, action, id):
    global x1_map, x2_map, x1_player, x2_player, y1, y2, step_index
    if step_index < DATABASE_SIZE:
        if id == 1:
            x1_map[step_index] = map
            x1_player[step_index] = player
            y1[step_index] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)
        if id == 2:
            x2_map[step_index] = map
            x2_player[step_index] = player
            y2[step_index] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)
