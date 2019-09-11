import tensorflow as tf
from constants import *
from RL import action_filter as AF
import time
import numpy as np


class Training():
    def __init__(self, m, c_m):
        self.model = m
        self.chat_model = c_m

        self.agents_grads = [self.model.trainable_variables] * 4
        self.chats_grads = [self.chat_model.trainable_variables] * 4

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=RL_LR)

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
        # 0.5 if it won but died, -0.5 if lost but didn't die first, else -1/1
        rewards = [(0.5 if a in died_first else 1) if a in won else -1 if a in died_first else -0.5 for a in range(4)]

        # if it is a tie, lower the punishment by 2 times
        rewards = [r if won else r / 2 for r in rewards]

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
        self.apply_grads(won, died_first)
        self.reset_grads()

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

    def end_step(self):
        for i in range(4):
            self.tapes[i] = [None] + self.tapes[i][:-1]
            self.messages[i] = [tf.zeros((1, 3, 3))] + self.messages[i][:-1]

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

    # just adds two sets of gradients together
    def add_grads(self, grads1, grads2):
        new_grads = [None] * len(grads1)
        for a in range(len(grads1)):
            if grads1[a] is None and grads2 is None:
                continue
            if grads1[a] is None:
                grads1[a] = np.zeros_like(grads2[a])
            if grads2[a] is None:
                grads2[a] = np.zeros_like(grads1[a])
            new_grads[a] = grads1[a] + grads2[a]
        return new_grads

    # adds gradients from pervious timesteps to gradients from this timestep
    def save_grads(self, agent_grads, chat_grads, id, step):

        for a in range(len(agent_grads)):
            # grads will be None at message / action layers
            if agent_grads[a] is None:
                continue
            self.agents_grads[id][a] = (self.agents_grads[id][a] * (step - 1) + agent_grads[a]) / step

        for m in range(len(chat_grads)):
            self.chats_grads[id][m] = (self.chats_grads[id][m] * (step - 1) + chat_grads[m]) / step

    def backprop_chat(self, chat_grads, model_grads, chat_model_grads, id):
        abs_grads = tf.abs(chat_grads)
        grads = tf.reduce_sum(abs_grads, [0, 1, 2])
        sizes, indexes = tf.math.top_k(grads, k=N_BP_MESSAGES)

        # al = 0
        # siz = 0.1
        for i in range(N_BP_MESSAGES):
            # if the message has even index, its this agent's, otherwise its his teammates
            msg_agent_id = (id + (indexes[i] % 2) * 2) % 4
            msg_id = indexes[i] // 2 + 1

            if self.tapes[msg_agent_id][msg_id] is None or sizes[i] < 1e-20:
                continue

            with self.tapes[msg_agent_id][msg_id]:
                # multiplied by an arbitrary number to combat vanishing gradient by upscaling everything to roughly
                # the same scale as the first model backprop
                msg = self.messages[msg_agent_id][msg_id] * chat_grads[:, :, :, indexes[i]] * 1e3

            m_g, c_m_g = self.tapes[msg_agent_id][msg_id].gradient(msg, [self.model.trainable_variables,
                                                                         self.chat_model.trainable_variables])

            # for i in range(len(m_g)):
            #     if m_g[i] is not None:
            #         al += np.sum(np.absolute(m_g[i]))
            #         siz += np.array(m_g[i]).size

            model_grads = self.add_grads(model_grads, m_g)
            chat_model_grads = self.add_grads(chat_model_grads, c_m_g)

        # print(al / siz)
        return model_grads, chat_model_grads

    def training_step(self, agent_features, id, step, position):
        tim = time.time()
        new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)

        chat = self.get_chat(id)
        a = time.time() - tim
        tim = time.time()
        with new_tape:
            new_tape.watch(self.model.trainable_variables)
            new_tape.watch(self.chat_model.trainable_variables)
            new_tape.watch(chat)
            b = time.time() - tim
            tim = time.time()

            chat_features = self.chat_model(chat)
            features = tf.concat([agent_features[:, :, :, :N_MAP_FEATURES], chat_features], 3)
            actions, msg = self.model(features)
            c = time.time() - tim
            self.add_message(msg, id)

            tim = time.time()
            action_filter = AF.get_action_filter(position[0], position[1], features)
            action = AF.apply_action_filter(action_filter, actions[0])
            d = time.time() - tim

            # if all actions result in a certain death, don't compute gradients and just return 0
            if action is None:
                self.tapes[id][0] = new_tape
                return 0
            tim = time.time()
            # computes loss in regards to the first possible choice of the model, that doesn't result in a certain death
            loss = self.compute_loss([action], actions[0])
        e = time.time() - tim
        tim = time.time()
        model_grads, chat_model_grads, chat_grads = new_tape.gradient(loss, [self.model.trainable_variables,
                                                                             self.chat_model.trainable_variables,
                                                                             chat])
        h = time.time() - tim
        # al = 0
        # siz = 0
        # for i in range(len(model_grads)):
        #     if model_grads[i] is not None:
        #         al += np.sum(np.absolute(model_grads[i]))
        #         siz += np.array(model_grads[i]).size
        # print(al / siz, "-------------------")

        tim = time.time()
        model_grads, chat_model_grads = self.backprop_chat(chat_grads, model_grads, chat_model_grads, id)
        f = time.time() - tim
        tim = time.time()
        self.save_grads(model_grads, chat_model_grads, id, step)

        self.tapes[id][0] = new_tape
        g = time.time() - tim
        #         print("all", round(a+b+c+d+e+f+g, 3), "get_vars", round(a, 3), "watch", round(b, 3), "forward", round(c, 3), "filter", round(d, 3), "losscomp+close", round(e, 3), "grads1", round(h, 3), "msggrads", round(f, 3), "addgrads", round(g, 3))

        return action
