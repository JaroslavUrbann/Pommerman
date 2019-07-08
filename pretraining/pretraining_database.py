import numpy as np
from constants import *
import tensorflow as tf

database_size = 0
step_index = 0

x1_map = np.zeros((database_size, 11, 11, N_MAP_FEATURES), dtype="float32")
x2_map = np.zeros((database_size, 11, 11, N_MAP_FEATURES), dtype="float32")
x1_player = np.zeros((database_size, 11, 11, N_PLAYER_FEATURES), dtype="float32")
x2_player = np.zeros((database_size, 11, 11, N_PLAYER_FEATURES), dtype="float32")

_default_y = np.zeros(N_CLASSES, dtype="float32")
_default_y[N_CLASSES - 1] = 1

y1 = np.tile(_default_y, (database_size, 1))
y2 = np.tile(_default_y, (database_size, 1))


def create_database(db_size):
    global x1_map, x2_map, x1_player, x2_player, y1, y2, step_index, database_size
    database_size = db_size
    step_index = 0
    x1_map = np.zeros((database_size, 11, 11, N_MAP_FEATURES), dtype="float32")
    x2_map = np.zeros((database_size, 11, 11, N_MAP_FEATURES), dtype="float32")
    x1_player = np.zeros((database_size, 11, 11, N_PLAYER_FEATURES), dtype="float32")
    x2_player = np.zeros((database_size, 11, 11, N_PLAYER_FEATURES), dtype="float32")

    _default_y = np.zeros(N_CLASSES, dtype="float32")
    _default_y[N_CLASSES - 1] = 1

    y1 = np.tile(_default_y, (database_size, 1))
    y2 = np.tile(_default_y, (database_size, 1))


def get_database():
    global x1_map, x2_map, x1_player, x2_player, y1, y2, step_index, database_size
    return [x1_map, x2_map, x1_player, x2_player], [y1, y2]


def add_data(map, player, action, id):
    global x1_map, x2_map, x1_player, x2_player, y1, y2, step_index
    i = step_index + int(id == 3 or id == 4)
    if i < database_size:
        if id == 1 or id == 3:
            x1_map[i] = map
            x1_player[i] = player
            y1[i] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)
        if id == 2 or id == 4:
            x2_map[i] = map
            x2_player[i] = player
            y2[i] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)