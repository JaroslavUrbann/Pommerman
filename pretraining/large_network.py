import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, Activation, BatchNormalization
from constants import *
import matplotlib.pyplot as plt
from pretraining import pretraining_database
import pandas


class LargeNetwork:

    def __init__(self, drive):
        self.name = None
        self.model_id = None
        self.log_id = None
        self.weights = None
        self.logs = None
        self.model = None
        self.drive = drive

    def load_model(self, model_id, log_id):
        self.model_id = model_id
        self.log_id = log_id
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

    # uploads new csv file if name is given, downloads old csv and appends new results if id is given
    def upload_logs(self, test_results=None):
        # if I want to append new data to an old log
        if self.log_id:
            self.logs = self.drive.CreateFile({'id': self.log_id})
            self.logs.GetContentFile(self.logs["title"])
            df = pandas.read_csv(self.logs["title"])

            # creates a new dataframe with new history and appends it to the old one
            df2 = pandas.DataFrame(self.history)

            # adds test results to the new dataframe
            df2["test"] = ""
            if test_results:
                df2.loc[df2.shape[0] - 1, 'test'] = str("/".join(str(i) for i in test_results))

            # adds n_samples to the new dataframe
            df2["n_samples"] = ""
            old_n_samples = int(df.loc[df.shape[0] - 1, 'n_samples'])
            df2["n_samples"] = df2.apply(lambda x: (x.name + 1) * self.n_samples + old_n_samples, axis=1)

            # appends new dataframe to the old one
            df = df.append(df2, ignore_index=True, sort=False)

            df.to_csv(self.logs["title"], index=False)
            self.logs.SetContentFile(self.logs["title"])

        # if I want to create a new model log
        if self.name:
            df = pandas.DataFrame(self.history)

            # creates a test column and writes the result if it is present
            df["test"] = ""
            if test_results:
                df.loc[df.shape[0] - 1, 'test'] = str("/".join(str(i) for i in test_results))

            # creates a n_samples column with the amount of samples trained on in total after each row
            df["n_samples"] = ""
            df["n_samples"] = df.apply(lambda x: (x.name + 1) * self.n_samples, axis=1)

            df.to_csv(self.name + ".csv", index=False)
            self.logs = self.drive.CreateFile({'title': self.name + ".csv"})
            self.logs.SetContentFile(self.name + ".csv")
        self.logs.Upload()

    def init_dummy_model(self, name):
        pass

    def init_model(self, name):
        self.name = name

        x1 = Input(shape=(11, 11, N_FEATURES))
        x2 = Input(shape=(11, 11, N_FEATURES))
        layer = tf.keras.layers.concatenate([x1, x2], axis=3)
        layer = Conv2D(128, 3, padding="same")(layer)
        layer = BatchNormalization()(layer)
        layer = Activation("relu")(layer)

        for _ in range(20):
            res = layer
            layer = Conv2D(128, 3, padding="same")(layer)
            layer = Activation("relu")(layer)
            layer = Conv2D(128, 3, padding="same")(layer)
            layer = BatchNormalization()(layer)
            layer = Add()([layer, res])
            layer = Activation("relu")(layer)

        flat = tf.keras.layers.Flatten()(layer)

        y1_fc1 = tf.keras.layers.Dense(1024, activation='relu')(flat)
        y1_fc2 = tf.keras.layers.Dense(512, activation='relu')(y1_fc1)
        y1_fc3 = tf.keras.layers.Dense(256, activation='relu')(y1_fc2)
        y1_fc4 = tf.keras.layers.Dense(64, activation='relu')(y1_fc3)
        y1_fc5 = tf.keras.layers.Dense(16, activation='relu')(y1_fc4)
        y1_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent1")(y1_fc5)

        y2_fc1 = tf.keras.layers.Dense(1024, activation='relu')(flat)
        y2_fc2 = tf.keras.layers.Dense(512, activation='relu')(y2_fc1)
        y2_fc3 = tf.keras.layers.Dense(256, activation='relu')(y2_fc2)
        y2_fc4 = tf.keras.layers.Dense(64, activation='relu')(y2_fc3)
        y2_fc5 = tf.keras.layers.Dense(16, activation='relu')(y2_fc4)
        y2_out = tf.keras.layers.Dense(N_CLASSES, activation='softmax', name="agent2")(y2_fc5)

        model = tf.keras.models.Model(inputs=[x1, x2], outputs=[y1_out, y2_out])
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LR), loss_weights=[1, 1],
                      metrics=['accuracy'])
        self.model = model

    def train_model_on_database(self, n_epochs):
        x, y = pretraining_database.get_database()
        self.n_samples = y[0].shape[0]
        self.history = self.model.fit(x, y, validation_split=0.1,
                                      epochs=n_epochs).history

    def predict(self, x1, x2):
        out1, out2 = self.model.predict([x1, x2])
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

    def plot_csv(self):
        plt.rcParams['figure.figsize'] = [15, 7]
        plt.rcParams.update({'font.size': 18})

        df = pandas.read_csv(self.logs["title"])

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='val_agent1_acc', ax=ax)
        df.plot(kind='line', x='n_samples', y='val_agent2_acc', ax=ax)
        plt.show()

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='agent1_acc', ax=ax)
        df.plot(kind='line', x='n_samples', y='agent2_acc', ax=ax)
        plt.show()

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='val_agent1_loss', ax=ax)
        df.plot(kind='line', x='n_samples', y='val_agent2_loss', ax=ax)
        df.plot(kind='line', x='n_samples', y='val_loss', ax=ax)
        plt.show()

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='agent1_loss', ax=ax)
        df.plot(kind='line', x='n_samples', y='agent2_loss', ax=ax)
        df.plot(kind='line', x='n_samples', y='loss', ax=ax)
        plt.show()


if __name__ == "__main__":
    LN = LargeNetwork(None)
    LN.init_model("sad")
