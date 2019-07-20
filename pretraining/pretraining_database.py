import numpy as np
from constants import *
import tensorflow as tf

database_size = 0
step_index = 0

x1 = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")
x2 = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")

_default_y = np.zeros(N_CLASSES, dtype="float32")
_default_y[N_CLASSES - 1] = 1

y1 = np.tile(_default_y, (database_size, 1))
y2 = np.tile(_default_y, (database_size, 1))


def create_database(db_size):
    global x1, x2, y1, y2, step_index, database_size
    database_size = db_size
    step_index = 0
    x1 = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")
    x2 = np.zeros((database_size, 11, 11, N_FEATURES), dtype="float32")

    _default_y = np.zeros(N_CLASSES, dtype="float32")
    _default_y[N_CLASSES - 1] = 1

    y1 = np.tile(_default_y, (database_size, 1))
    y2 = np.tile(_default_y, (database_size, 1))


def get_database():
    global x1, x2, y1, y2
    return [x1, x2], [y1, y2]


def add_data(features, action, id):
    global x1, x2, y1, y2, step_index
    i = step_index + int(id == 3 or id == 4)
    if i < database_size:
        if id == 1 or id == 3:
            x1[i] = features
            y1[i] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)
        if id == 2 or id == 4:
            x2[i] = features
            y2[i] = tf.keras.utils.to_categorical(action, num_classes=N_CLASSES)
