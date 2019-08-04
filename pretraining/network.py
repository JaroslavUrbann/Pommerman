import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, Activation, BatchNormalization
from tensorflow.python.keras.regularizers import l2
from constants import *
import matplotlib.pyplot as plt
from pretraining import DB
import pandas


class Network:

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
    def upload_logs(self):
        # if I want to append new data to an old log
        if self.log_id:
            self.logs = self.drive.CreateFile({'id': self.log_id})
            self.logs.GetContentFile(self.logs["title"])
            df = pandas.read_csv(self.logs["title"])

            # creates a new dataframe with new history and appends it to the old one
            df2 = pandas.DataFrame(self.history)

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
        l2const = 1e-4

        x = Input(shape=(11, 11, N_FEATURES))
        layer = Conv2D(256, 3, padding="same", kernel_regularizer=l2(l2const))(x)
        layer = Activation("relu")(layer)
        layer = BatchNormalization()(layer)

        for _ in range(40):
            res = layer
            layer = Conv2D(256, 3, padding="same", kernel_regularizer=l2(l2const))(layer)
            layer = Activation("relu")(layer)
            layer = BatchNormalization()(layer)
            layer = Conv2D(256, 3, padding="same", kernel_regularizer=l2(l2const))(layer)
            layer = BatchNormalization()(layer)
            layer = Add()([layer, res])
            layer = Activation("relu")(layer)

        y = Conv2D(128, 1, padding="same", kernel_regularizer=l2(l2const))(layer)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)
        y = Flatten()(y)
        y = Dense(1024, kernel_regularizer=l2(l2const))(y)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)

        y = Dense(512, kernel_regularizer=l2(l2const))(y)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)

        y = Dense(256, kernel_regularizer=l2(l2const))(y)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)

        y = Dense(64, kernel_regularizer=l2(l2const))(y)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)

        y = Dense(16, activation="relu")(y)
        y = Dense(N_CLASSES, activation='softmax', name="y")(y)

        message = Conv2D(128, 1, padding="same", kernel_regularizer=l2(l2const))(layer)
        message = Activation("relu")(message)
        message = BatchNormalization()(message)
        message = Flatten()(message)
        message = Dense(1024, kernel_regularizer=l2(l2const))(message)
        message = Activation("relu")(message)
        message = BatchNormalization()(message)

        message = Dense(512, kernel_regularizer=l2(l2const))(message)
        message = Activation("relu")(message)
        message = BatchNormalization()(message)

        message = Dense(256, kernel_regularizer=l2(l2const))(message)
        message = Activation("relu")(message)
        message = BatchNormalization()(message)

        message = Dense(64, kernel_regularizer=l2(l2const))(message)
        message = Activation("relu")(message)
        message = BatchNormalization()(message)

        message = Dense(16, activation="relu")(message)
        message = Dense(N_MESSAGE_BITS, activation='tanh', name="message")(message)

        model = tf.keras.models.Model(inputs=x, outputs=[y, message])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LR),
                      metrics=['accuracy'],
                      loss_weights=[1., 0.])
        self.model = model

    def train_model_on_database(self, n_epochs):
        x, y = DB.get_database()
        self.n_samples = y.shape[0]
        self.history = self.model.fit(x, [y, y], validation_split=0.1,
                                      epochs=n_epochs).history

    def predict(self, x):
        return self.model.predict(x)

    def plot_csv(self):
        plt.rcParams['figure.figsize'] = [15, 7]
        plt.rcParams.update({'font.size': 18})

        df = pandas.read_csv(self.logs["title"])

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='val_y_acc', ax=ax)
        df.plot(kind='line', x='n_samples', y='y_acc', ax=ax)
        plt.show()

        ax = plt.gca()
        df.plot(kind='line', x='n_samples', y='val_y_loss', ax=ax)
        df.plot(kind='line', x='n_samples', y='y_loss', ax=ax)
        plt.show()
