import tensorflow as tf
from constants import *
from feature_engineer import FeatureEngineer


@tf.custom_gradient
def half_grad(x):
    def _grad(dy):
        return dy * .5
    return x, _grad


class RLTraining:

    def __init__(self, model, message_model):
        self.model = model
        self.message_model = message_model
        self.tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        self.feature_engineers = [FeatureEngineer() for _ in range(4)]
        self.messages = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(MESSAGE_HISTORY_LENGTH)] for _ in range(4)]
        self.next_messages = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
        with self.tape:
            self.tape.watch(self.model.trainable_variables)
            self.tape.watch(self.message_model.trainable_variables)

    def next_time_step(self):
        for id in range(4):
            self.messages[id] = self.messages[id][-1:] + self.messages[id][:-1]
            self.messages[id][0] = [self.next_messages[id]]
        self.next_messages = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]

    def training_step(self, observation, id):

        # get features where last few layers are 0s representing message features
        FE = self.feature_engineers[id]
        agent_features = FE.get_features(observation)

        # takes messages from his teammate
        messages = self.messages[(id + 2) % 4]
        messages = tf.concat(messages, 3)

        # gets message features to put in his network
        message_features = self.message_model(messages)

        features = tf.concat([agent_features[:21], message_features], 2)
        action, message = self.model(tf.expand_dims(features, 0))

        # reshape message and add it to stack
        message = tf.reshape(message, (1, 2, 3, 1))
        padding = tf.zeros((1, 1, 3, 1))
        message = tf.concat[message, padding]
        self.next_messages[id] = message
