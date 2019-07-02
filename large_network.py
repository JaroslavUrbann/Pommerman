import tensorflow as tf
import numpy as np
import time


class LargeNetwork:

    n_classes = 12
    lr = 3e-4
    n_epochs = 1
    n_batches = 3
    batch_size = 4
    n_map_features = 14
    n_player_features = 1

    def initialize_model(self):
        x1_map = tf.keras.layers.Input(shape=(11, 11, self.n_map_features))
        x2_map = tf.keras.layers.Input(shape=(11, 11, self.n_map_features))
        x1_player = tf.keras.layers.Input(shape=(11, 11, self.n_player_features))
        x2_player = tf.keras.layers.Input(shape=(11, 11, self.n_player_features))

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

        fc1 = tf.keras.layers.Dense(4096, activation='relu')(flat)
        fc2 = tf.keras.layers.Dense(1024, activation='relu')(fc1)
        fc3 = tf.keras.layers.Dense(512, activation='relu')(fc2)
        fc4 = tf.keras.layers.Dense(128, activation='relu')(fc3)
        fc5 = tf.keras.layers.Dense(32, activation='relu')(fc4)
        out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(fc5)

        model = tf.keras.models.Model(inputs=[x1_map, x2_map, x1_player, x2_player], outputs=out)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        self.model = model
        model.summary()

    def train_model(self):
        x1_map = np.ones(shape=(512, 11, 11, self.n_map_features))
        x2_map = np.ones(shape=(512, 11, 11, self.n_map_features))
        x1_player = np.ones(shape=(512, 11, 11, self.n_player_features))
        x2_player = np.ones(shape=(512, 11, 11, self.n_player_features))
        y = np.ones(shape=(512, self.n_classes))
        self.model.fit([x1_map, x2_map, x1_player, x2_player], y, epochs=10)

    def test_model(self):
        x1_map = np.random.rand(1, 11, 11, self.n_map_features).astype("float32")
        x2_map = np.random.rand(1, 11, 11, self.n_map_features).astype("float32")
        x1_player = np.random.rand(1, 11, 11, self.n_player_features).astype("float32")
        x2_player = np.random.rand(1, 11, 11, self.n_player_features).astype("float32")
        tim = time.time()
        print(self.model([x1_map, x2_map, x1_player, x2_player]))
        print(time.time() - tim)


LN = LargeNetwork()
LN.initialize_model()
LN.train_model()
# LN.test_model()