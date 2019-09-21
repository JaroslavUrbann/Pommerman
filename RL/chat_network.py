import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, Activation, BatchNormalization, UpSampling2D, Cropping2D
from tensorflow.python.keras.regularizers import l2
from constants import *


class ChatNetwork:

    def __init__(self, drive, name=None, model_id=None):
        self.name = name
        self.model_id = None if model_id is None else model_id[33:]
        self.weights = None
        self.model = None
        self.drive = drive

    def load_model(self):
        if self.model_id is not None:
            self.weights = self.drive.CreateFile({'id': self.model_id})
            self.weights.GetContentFile(self.weights["title"])
            self.model = tf.keras.models.load_model(self.weights["title"])
        else:
            self.init_model()

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
        x = Input(shape=(3, 3, CHAT_HISTORY_LENGTH))
        # 3 x 3 x 128
        layer = Conv2D(128, 3, padding="same", activation="relu")(x)
        layer = UpSampling2D()(layer)
        # 6 x 6 x 32
        layer = Conv2D(32, 3, padding="same", activation="relu")(layer)
        layer = UpSampling2D()(layer)
        layer = Cropping2D(cropping=((0, 1), (0, 1)))(layer)
        # 11 x 11 x 8
        layer = Conv2D(8, 3, padding="same", activation="relu")(layer)
        y = Conv2D(4, 3, padding="same", activation="relu")(layer)

        model = tf.keras.models.Model(inputs=x, outputs=y)
        self.model = model

    def predict(self, x):
        return self.model(x)
