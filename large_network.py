import tensorflow as tf
import numpy as np
import time
from constants import *
import matplotlib.pyplot as plt
import pretraining_database
import pandas


class LargeNetwork:

    def __init__(self, drive):
        self.name = None
        self.model_id = None
        self.log_id = None
        self.weights = None
        self.model = None
        self.drive = drive

    def load_model(self, model_id, log_id):
        self.model_id = model_id
        self.log_id = log_id
        self.weights = self.drive.CreateFile({'id': self.model_id})
        self.weights.GetContentFile(self.weights["title"] + ".h5")
        self.model = tf.keras.models.load_model(self.weights["title"] + ".h5")

    # uploads new weights if name is given, otherwise updates weights that were downloaded in load_model
    def upload_model(self):
        if self.name:
            self.model.save(self.name + ".h5", overwrite=True)
            new_model = self.drive.CreateFile({'title': self.name + ".h5"})
            new_model.SetContentFile(self.name + ".h5")
            new_model.Upload()
        else:
            self.model.save(self.weights["title"] + ".h5", overwrite=True)
            self.weights.SetContentFile(self.weights["title"] + ".h5")
            self.weights.Upload()

    # uploads new csv file if name is given, downloads old csv and appends new results if id is given
    def upload_logs(self, test_results=None):
        if self.log_id:
            logs = self.drive.CreateFile({'id': self.log_id})
            logs.GetContentFile(logs["title"] + ".csv")
            df = pandas.read_csv(logs["title"] + ".csv")
            df2 = pandas.DataFrame(self.history)
            df2["test"] = ""
            df = df.append(df2, ignore_index=True, sort=False)
            if test_results:
                n_rows, n_cols = df.shape
                df.iat[n_rows - 1, n_cols - 1] = str("/".join(str(i) for i in test_results))
            df.to_csv(logs["title"] + ".csv", index=False)
            logs.SetContentFile(logs["title"] + ".csv")
        if self.name:
            df = pandas.DataFrame(self.history)
            df["test"] = ""
            if test_results:
                n_rows, n_cols = df.shape
                df.iat[n_rows - 1, n_cols - 1] = str("/".join(str(i) for i in test_results))
            df.to_csv(self.name + ".csv", index=False)
            logs = self.drive.CreateFile({'title': self.name + ".csv"})
            logs.SetContentFile(self.name + ".csv")
        logs.Upload()

    def init_dummy_model(self, name):
        self.name = name

        x1_map = tf.keras.layers.Input(shape=(11, 11, N_MAP_FEATURES))
        x2_map = tf.keras.layers.Input(shape=(11, 11, N_MAP_FEATURES))
        x1_player = tf.keras.layers.Input(shape=(11, 11, N_PLAYER_FEATURES))
        x2_player = tf.keras.layers.Input(shape=(11, 11, N_PLAYER_FEATURES))

        conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")

        x1_latent = conv1(x1_map)
        x2_latent = conv1(x2_map)

        merged = tf.keras.layers.concatenate([x1_latent, x1_player, x2_latent, x2_player], axis=3)

        conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(merged)

        flat = tf.keras.layers.Flatten()(conv2)

        y1_fc1 = tf.keras.layers.Dense(32, activation='relu')(flat)
        y1_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent1")(y1_fc1)

        y2_fc1 = tf.keras.layers.Dense(32, activation='relu')(flat)
        y2_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent2")(y2_fc1)

        model = tf.keras.models.Model(inputs=[x1_map, x2_map, x1_player, x2_player], outputs=[y1_out, y2_out])
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LR), loss_weights=[1, 1],
                      metrics=['accuracy'])
        self.model = model

    def init_model(self, name):
        self.name = name

        x1_map = tf.keras.layers.Input(shape=(11, 11, N_MAP_FEATURES))
        x2_map = tf.keras.layers.Input(shape=(11, 11, N_MAP_FEATURES))
        x1_player = tf.keras.layers.Input(shape=(11, 11, N_PLAYER_FEATURES))
        x2_player = tf.keras.layers.Input(shape=(11, 11, N_PLAYER_FEATURES))

        conv1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")
        conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")
        conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")
        conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")
        conv6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")

        x1_latent = conv6(conv5(conv4(conv3(conv2(conv1(x1_map))))))
        x2_latent = conv6(conv5(conv4(conv3(conv2(conv1(x2_map))))))

        merged = tf.keras.layers.concatenate([x1_latent, x1_player, x2_latent, x2_player], axis=3)

        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(merged)
        conv8 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(conv7)
        conv9 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(conv8)
        conv10 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(conv9)
        conv11 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same")(conv10)

        flat = tf.keras.layers.Flatten()(conv11)

        y1_fc1 = tf.keras.layers.Dense(2048, activation='relu')(flat)
        y1_fc2 = tf.keras.layers.Dense(512, activation='relu')(y1_fc1)
        y1_fc3 = tf.keras.layers.Dense(256, activation='relu')(y1_fc2)
        y1_fc4 = tf.keras.layers.Dense(64, activation='relu')(y1_fc3)
        y1_fc5 = tf.keras.layers.Dense(16, activation='relu')(y1_fc4)
        y1_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent1")(y1_fc5)

        y2_fc1 = tf.keras.layers.Dense(2048, activation='relu')(flat)
        y2_fc2 = tf.keras.layers.Dense(512, activation='relu')(y2_fc1)
        y2_fc3 = tf.keras.layers.Dense(256, activation='relu')(y2_fc2)
        y2_fc4 = tf.keras.layers.Dense(64, activation='relu')(y2_fc3)
        y2_fc5 = tf.keras.layers.Dense(16, activation='relu')(y2_fc4)
        y2_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent2")(y2_fc5)

        model = tf.keras.models.Model(inputs=[x1_map, x2_map, x1_player, x2_player], outputs=[y1_out, y2_out])
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LR), loss_weights=[1, 1],
                      metrics=['accuracy'])
        self.model = model

    def train_model_on_database(self, n_epochs):
        x, y = pretraining_database.get_database()
        self.history = self.model.fit(x, y, validation_split=0.1,
                                      epochs=n_epochs).history

    def predict(self, x1_map, x2_map, x1_player, x2_player):
        out1, out2 = self.model([x1_map, x2_map, x1_player, x2_player])
        return out1, out2

    def plot_history(self):
        # Plot training & validation accuracy values
        plt.plot(self.history['agent1_acc'])
        plt.plot(self.history['val_agent1_acc'])
        plt.plot(self.history['agent2_acc'])
        plt.plot(self.history['val_agent2_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['agent1_acc', 'val_agent1_acc', 'agent2_acc', 'val_agent2_acc'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history['agent1_loss'])
        plt.plot(self.history['val_agent1_loss'])
        plt.plot(self.history['agent2_loss'])
        plt.plot(self.history['val_agent2_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['agent1_loss', 'val_agent1_loss', 'agent2_loss', 'val_agent2_loss'], loc='upper left')
        plt.show()


if __name__ == "__main__":
    LN = LargeNetwork()
    LN.init_dummy_model()
    pretraining_database.create_database(320)
    LN.train_model_on_database(4)
