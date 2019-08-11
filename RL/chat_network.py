import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, Activation, BatchNormalization, UpSampling2D
from tensorflow.python.keras.regularizers import l2
from constants import *


class ChatNetwork:

    def __init__(self, drive, name=None, model_id=None):
        self.name = name
        self.model_id = model_id
        self.weights = None
        self.model = None
        self.drive = drive

    def load_model(self):
        self.weights = self.drive.CreateFile({'id': self.model_id})
        self.weights.GetContentFile(self.weights["title"])
        self.model = tf.keras.models.load_model(self.weights["title"])

    # uploads new weights if name is given, otherwise updates weights that were downloaded in load_model
    def upload_model(self):
        if self.name:
            self.model.save(self.name + ".h5", overwrite=True)
            new_model = self.drive.CreateFile({'title': self.name + ".h5"})
            new_model.SetContentFile(self.name + ".h5")
            new_model.Upload()
        else:
            self.model.save(self.weights["title"], overwrite=True)
            self.weights.SetContentFile(self.weights["title"])
            self.weights.Upload()

    def init_model(self):
        l2const = 1e-4

        x = Input(shape=(3, 3, CHAT_HISTORY_LENGTH))
        layer = Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(x)
        layer = Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(layer)
        layer = UpSampling2D(layer)
        layer = Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(layer)
        layer = Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(layer)
        layer = UpSampling2D(layer)
        layer = Conv2D(64, 3, padding="valid", activation="relu", kernel_regularizer=l2(l2const))(layer)
        layer = Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(layer)
        y = Conv2D(4, 3, padding="same", activation="relu", kernel_regularizer=l2(l2const))(layer)

        model = tf.keras.models.Model(inputs=x, outputs=y)
        self.model = model

    def predict(self, x):
        return self.model(x)
