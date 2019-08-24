import tensorflow as tf
from constants import *
import time

tf.enable_eager_execution()


class RLTraining:


    def __init__(self, model, chat_model):
        self.model = model
        self.chat_model = chat_model

        self.agents_grads = [model.trainable_variables] * 4
        self.chats_grads = [chat_model.trainable_variables] * 4

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        self.reset_grads()


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


    # died_first should ideally have at most 1 player from each team
    def apply_grads(self, won, died_first):
        # 0.7 if it won but died, -0.7 if lost but didn't die first, else -1/1
        rewards = [(0.7 if a in died_first else 1) if a in won else -1 if a in died_first else -0.7 for a in range(4)]

        for var in range(len(self.agents_grads[0])):
            grads_agent_0 = self.agents_grads[0][var] * rewards[0]
            grads_agent_1 = self.agents_grads[1][var] * rewards[1]
            grads_agent_2 = self.agents_grads[2][var] * rewards[2]
            grads_agent_3 = self.agents_grads[3][var] * rewards[3]
            self.agents_grads[0][var] = (grads_agent_0 + grads_agent_1 + grads_agent_2 + grads_agent_3) / 4

        for var in range(len(self.chats_grads[0])):
            grads_msgs_0 = self.chats_grads[0][var] * rewards[0]
            grads_msgs_1 = self.chats_grads[1][var] * rewards[1]
            grads_msgs_2 = self.chats_grads[2][var] * rewards[2]
            grads_msgs_3 = self.chats_grads[3][var] * rewards[3]
            self.chats_grads[0][var] = (grads_msgs_0 + grads_msgs_1 + grads_msgs_2 + grads_msgs_3) / 4

        self.optimizer.apply_gradients(zip(self.agents_grads[0], self.model.trainable_variables))
        self.optimizer.apply_gradients(zip(self.chats_grads[0], self.chat_model.trainable_variables))


    def end_episode(self, won, died_first):
        if won:
            self.apply_grads(won, died_first)
        self.reset_grads()

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]


    def end_step(self):
        for i in range(4):
            self.tapes[i] = self.tapes[i][-1:] + self.tapes[i][:-1]
            self.messages[i] = self.messages[i][-1:] + self.messages[i][:-1]


    def get_chat(self, id):
        chat = [None] * CHAT_HISTORY_LENGTH
        chat[0::2] = self.messages[id][1:]
        chat[1::2] = self.messages[(id + 2) % 4][1:]
        chat = tf.stack(chat, 3)
        return chat


    def add_message(self, msg, id):
        # add noise and change to 0 - 1 distribution
        msg += tf.random.normal(msg.shape, mean=0.0, stddev=10.0)
        msg = tf.math.sigmoid(msg)

        # reshape msg and add it to stack
        msg = tf.reshape(msg, (1, 2, 3))
        padding = tf.zeros((1, 1, 3))
        msg = tf.concat([msg, padding], 1)
        self.messages[id][0] = msg


    # adds gradients from pervious timesteps to gradients from this timestep
    # how much weight is given to gradients from previous timesteps depends on GRADIENT_DISCOUNT
    def add_grads(self, agent_grads, chat_grads, id, step):
        # equation for first n elements of a geometric sequence
        total_avg = (1 - GRADIENT_DISCOUNT ** (step + 1)) / (1 - GRADIENT_DISCOUNT)

        for a in range(len(self.agents_grads[0])):
            # grads will be None at message layers
            if agent_grads[a] is None:
                continue
            self.agents_grads[id][a] = (self.agents_grads[id][a] * (total_avg - 1) + agent_grads[a]) / total_avg

        # are later chats really more important? questionable decision
        for m in range(len(self.chats_grads[0])):
            self.chats_grads[id][m] = (self.chats_grads[id][m] * (total_avg - 1) + chat_grads[m]) / total_avg


    def backprop_chat(self, chat_grads, model_grads, chat_model_grads, id):
        abs_grads = tf.abs(chat_grads)
        grads = tf.reduce_sum(abs_grads, [0, 1, 2])
        sizes, indexes = tf.math.top_k(grads, k=N_BP_MESSAGES)

        for i in range(N_BP_MESSAGES):
            # if the message has even index, its this agent's, otherwise its his teammates
            msg_agent_id = (id + (indexes[i] % 2) * 2) % 4
            msg_id = indexes[i] // 2 + 1

            if self.tapes[msg_agent_id][msg_id] is None or sizes[i] == 0.0:
                continue

            with self.tapes[msg_agent_id][msg_id]:
                self.messages[msg_agent_id][msg_id] *= chat_grads[:, :, :, indexes[i]] * 0.5 / N_BP_MESSAGES
            m_g, c_m_g = self.tapes[msg_agent_id][msg_id].gradient(self.messages[msg_agent_id][msg_id],
                                                                   [self.model.trainable_variables,
                                                                    self.chat_model.trainable_variables])
            model_grads += m_g
            chat_model_grads += c_m_g

        return model_grads, chat_model_grads


    def training_step(self, agent_features, id, step):
        new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)

        chat = self.get_chat(id)

        with new_tape:
            new_tape.watch(self.model.trainable_variables)
            new_tape.watch(self.chat_model.trainable_variables)
            new_tape.watch(chat)

            # gets chat features to put in his network
            chat_features = self.chat_model(chat)
            features = tf.concat([agent_features[:, :, :, :21], chat_features], 3)
            actions, msg = self.model(features)

            self.add_message(msg, id)

            action = tf.math.argmax(actions[0])
            loss = self.compute_loss([action], actions[0]) / 2

        model_grads, chat_model_grads, chat_grads = new_tape.gradient(loss, [self.model.trainable_variables,
                                                                             self.chat_model.trainable_variables, chat])
        new_model_grads, new_chat_model_grads = self.backprop_chat(chat_grads, model_grads, chat_model_grads, id)

        self.add_grads(new_model_grads, new_chat_model_grads, id, step)

        self.tapes[id][0] = new_tape

        return action
