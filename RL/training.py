import tensorflow as tf
from constants import *
from feature_engineer import FeatureEngineer


class RLTraining:

    def __init__(self, model, chat_model):
        self.model = model
        self.chat_model = chat_model
        self.agents_grads = [model.trainable_variables for _ in range(4)]
        self.chats_grads = [chat_model.trainable_variables for _ in range(4)]
        self.tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        self.feature_engineers = [FeatureEngineer() for _ in range(4)]
        self.chats = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(CHAT_HISTORY_LENGTH)] for _ in range(2)]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.reset_grads()
        with self.tape:
            self.tape.watch(model.trainable_variables)
            self.tape.watch(chat_model.trainable_variables)

    def reset_grads(self):
        for i in range(len(self.agents_grads[0])):
            self.agents_grads[0][i] = self.agents_grads[0][i] * 0
            self.agents_grads[1][i] = self.agents_grads[1][i] * 0
            self.agents_grads[2][i] = self.agents_grads[2][i] * 0
            self.agents_grads[3][i] = self.agents_grads[3][i] * 0
        for i in range(len(self.chats_grads[0])):
            self.chats_grads[0][i] = self.chats_grads[0][i] * 0
            self.chats_grads[1][i] = self.chats_grads[1][i] * 0
            self.chats_grads[2][i] = self.chats_grads[2][i] * 0
            self.chats_grads[3][i] = self.chats_grads[3][i] * 0

    def add_grads(self, agent_grads, chat_grads, id, step):
        # equation for first n elements of a geometric sequence
        total_discount = (1 - GRADIENT_DISCOUNT**(step + 1)) / (1 - GRADIENT_DISCOUNT)
        for a in range(len(self.agents_grads[0])):
            new_avg = (self.agents_grads[id][a] * (total_discount - 1) + agent_grads[a]) / total_discount
            self.agents_grads[id][a] = new_avg

        # are later chats really more important? questionable decision
        for m in range(len(self.chats_grads[0])):
            new_avg = (self.chats_grads[id][m] * (total_discount - 1) + chat_grads[m]) / total_discount
            self.chats_grads[id][m] = new_avg

    # died_first needs to have at most 1 player from each team
    def apply_grads(self, won, died_first):
        # 0.7 if it won but died, -0.7 if lost but didn't die first, else -1/1
        rewards = [(0.7 if a in died_first else 1) if a in won else -1 if a in died_first else -0.7 for a in range(4)]
        for var in range(len(self.agents_grads[0])):
            grads_agent_0 = self.agents_grads[0][var] * rewards[0]
            grads_agent_1 = self.agents_grads[1][var] * rewards[1]
            grads_agent_2 = self.agents_grads[2][var] * rewards[2]
            grads_agent_3 = self.agents_grads[3][var] * rewards[3]
            self.agents_grads[0][var] = (grads_agent_0 + grads_agent_1 + grads_agent_2 + grads_agent_3) / 4
        self.optimizer.apply_grads(zip(self.agents_grads[0], self.model.trainable_variables))

        for var in range(len(self.chats_grads[0])):
            grads_msgs_0 = self.chats_grads[0][var] * rewards[0]
            grads_msgs_1 = self.chats_grads[1][var] * rewards[1]
            grads_msgs_2 = self.chats_grads[2][var] * rewards[2]
            grads_msgs_3 = self.chats_grads[3][var] * rewards[3]
            self.chats_grads[0][var] = (grads_msgs_0 + grads_msgs_1 + grads_msgs_2 + grads_msgs_3) / 4
        self.optimizer.apply_grads(zip(self.chats_grads[0], self.chat_model.trainable_variables))

    def end_episode(self, won, died_first):
        if won:
            self.apply_grads(won, died_first)
        self.reset_grads()
        self.feature_engineers = [FeatureEngineer() for _ in range(4)]
        self.chats = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(CHAT_HISTORY_LENGTH)] for _ in range(2)]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
        self.tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        with self.tape:
            self.tape.watch(self.model.trainable_variables)
            self.tape.watch(self.chat_model.trainable_variables)

    # exchanges messages, is needed because some agents might be dead and can't add theirs to the chat
    def next_time_step(self):
        for i in range(2):
            self.chats[i] = self.chats[i][-2:] + self.chats[i][:-2]
            self.chats[i][0] = [self.next_msgs[i]]
            self.chats[i][1] = [self.next_msgs[i + 2]]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]

    @tf.custom_gradient
    def half_grad(self, x):
        def _grad(dy):
            return dy / CHAT_HISTORY_LENGTH  # dy * 0.5 / (CHL/2)

        return x, _grad

    def training_step(self, observation, id):
        with self.tape:
            # get features where last few layers are 0s representing chat features
            FE = self.feature_engineers[id]
            agent_features = FE.get_features(observation)

            # takes chat conversation
            chat = self.chats[id % 2]
            chat = tf.concat(chat, 3)

            # gets chat features to put in his network
            chat_features = self.chat_model(chat)

            features = tf.concat([agent_features[:21], chat_features], 2)
            actions, msg = self.model(tf.expand_dims(features, 0))

            # halve msg gradient
            m = self.half_grad(msg)

            # reshape msg and add it to stack
            msg = tf.reshape(m, (1, 2, 3, 1))
            padding = tf.zeros((1, 1, 3, 1))
            m_ = tf.concat[msg, padding]
            self.next_msgs[id] = m_

            action = tf.math.argmax(actions)
            loss = self.compute_loss([action], actions)

            agent_grads = self.tape.gradient(loss, self.model.trainable_variables)
            chat_grads = self.tape.gradient(loss, self.chat_model.trainable_variables)

            self.add_grads(agent_grads, chat_grads, id, int(observation["step_count"]))