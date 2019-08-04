import tensorflow as tf
from constants import *
from feature_engineer import FeatureEngineer


class RLTraining:

    def __init__(self, model, message_model):
        self.model = model
        self.message_model = message_model
        self.tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        self.feature_engineers = [FeatureEngineer() for _ in range(4)]
        self.messages = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(CHAT_HISTORY_LENGTH)] for _ in range(2)]
        self.next_messages = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        with self.tape:
            self.tape.watch(self.model.trainable_variables)
            self.tape.watch(self.message_model.trainable_variables)

    def next_time_step(self):
        for i in range(2):
            self.messages[i] = self.messages[i][-2:] + self.messages[i][:-2]
            self.messages[i][0] = [self.next_messages[i]]
            self.messages[i][1] = [self.next_messages[i + 2]]
        self.next_messages = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]

    @tf.custom_gradient
    def half_grad(self, x):
        def _grad(dy):
            return dy / CHAT_HISTORY_LENGTH  # dy * 0.5 / (CHL/2)

        return x, _grad

    def training_step(self, observation, id):
        with self.tape:
            # get features where last few layers are 0s representing message features
            FE = self.feature_engineers[id]
            agent_features = FE.get_features(observation)

            # takes message conversation
            messages = self.messages[id % 2]
            msgs = tf.concat(messages, 3)

            # gets message features to put in his network
            message_features = self.message_model(msgs)

            features = tf.concat([agent_features[:21], message_features], 2)
            actions, message = self.model(tf.expand_dims(features, 0))

            # halve message gradient
            m = self.half_grad(message)

            # reshape message and add it to stack
            msg = tf.reshape(m, (1, 2, 3, 1))
            padding = tf.zeros((1, 1, 3, 1))
            m_ = tf.concat[msg, padding]
            self.next_messages[id] = m_

            action = tf.math.argmax(actions)
            loss = self.compute_loss([action], actions)
