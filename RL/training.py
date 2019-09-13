import tensorflow as tf
from constants import *
from RL import action_filter as AF
import time
import numpy as np


class Training:
    def __init__(self, m, c_m):
        self.models = m
        self.chat_model = c_m
        self.step = 0

        # (3,4,whatever)
        # three models have different gradients for
        # four different agents, that have
        # whatever amount of layers
        self.models_grads = [[self.models[0].trainable_variables] * 4,
                             [self.models[1].trainable_variables] * 4,
                             [self.models[2].trainable_variables] * 4]
        self.chats_grads = [self.chat_model.trainable_variables] * 4
        # track number of updates for each model and each agent (to make an average afterwards)
        self.n_grad_updates = [[0] * 4 for _ in range(3)]

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # 4 different optimizer instances for 4 different models
        self.optimizers = [tf.keras.optimizers.Adam(learning_rate=RL_LR)] * 4

        self.reset_grads()


    def reset_grads(self):
        for m in range(len(self.models_grads)):
            for a in range(len(self.models_grads[m])):
                for l in range(len(self.models_grads[m][a])):
                    self.models_grads[m][a][l] *= 0
        for a in range(len(self.chats_grads)):
            for l in range(len(self.chats_grads[a])):
                self.chats_grads[a][l] *= 0


    # died_first should ideally have at most 1 player from each team
    def apply_grads(self, won, died_first):
        # 0.5 if it won but died, -0.5 if lost but didn't die first, else -1/1
        rewards = [(0.5 if a in died_first else 1) if a in won else -1 if a in died_first else -0.5 for a in range(4)]

        # if it is a tie, lower the punishment by 2 times
        rewards = [r if won else r / 2 for r in rewards]

        for m in range(len(self.models_grads)):
            model_grads = []
            for l in range(len(self.models_grads[m][0])):
                layer_grads = np.zeros_like(self.models_grads[m][0][l])
                for a in range(len(self.models_grads[m])):
                    if self.n_grad_updates[m][a] > 0:
                        layer_grads += self.models_grads[m][a][l] * rewards[a] * 0.25 / self.n_grad_updates[m][a]
                model_grads.append(layer_grads)
            self.optimizers[m].apply_gradients(zip(model_grads, self.models[m].trainable_variables))

        chat_model_grads = []
        n_grad_updates = np.sum(self.n_grad_updates, axis=0)
        for l in range(len(self.chats_grads[0])):
            layer_grads = np.zeros_like(self.chats_grads[0][l])
            for a in range(len(self.chats_grads)):
                layer_grads += self.chats_grads[a][l] * rewards[a] * 0.25 / n_grad_updates[a]
            chat_model_grads.append(layer_grads)
        self.optimizers[3].apply_gradients(zip(chat_model_grads, self.chat_model.trainable_variables))


    def end_episode(self, won, died_first):
        self.apply_grads(won, died_first)
        self.reset_grads()

        # this timestep + 32 previous timesteps
        n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
        self.tapes = [[None] * n_msgs for _ in range(4)]
        self.messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]
        self.step = 0
        self.n_grad_updates = [[0] * 4 for _ in range(3)]


    def end_step(self):
        self.step += 1
        for i in range(4):
            self.tapes[i] = [None] + self.tapes[i][:-1]
            self.messages[i] = [tf.zeros((1, 3, 3))] + self.messages[i][:-1]


    def get_chat(self, a_id):
        chat = [None] * CHAT_HISTORY_LENGTH
        chat[0::2] = self.messages[a_id][1:]
        chat[1::2] = self.messages[(a_id + 2) % 4][1:]
        chat = tf.stack(chat, 3)
        return chat


    def add_message(self, msg, a_id):
        # add noise and change to 0 - 1 distribution
        msg += tf.random.normal(msg.shape, mean=0.0, stddev=10.0)
        msg = tf.math.sigmoid(msg)

        # reshape msg and add it to stack
        msg = tf.reshape(msg, (1, 2, 3))
        padding = tf.zeros((1, 1, 3))
        msg = tf.concat([msg, padding], 1)
        self.messages[a_id][0] = msg


    # adds gradients from pervious timesteps to gradients from this timestep
    def save_grads(self, agent_grads, chat_grads, m_i, a_id, weight):

        self.n_grad_updates[m_i][a_id] += weight

        for l in range(len(agent_grads)):
            # grads will be None at message / action layers
            if agent_grads[l] is None:
                continue
            self.models_grads[m_i][a_id][l] += agent_grads[l]

        for l in range(len(chat_grads)):
            self.chats_grads[a_id][l] += chat_grads[l]


    def backprop_chat(self, chat_grads, id):
        abs_grads = tf.abs(chat_grads)
        grads = tf.reduce_sum(abs_grads, [0, 1, 2])
        sizes, indexes = tf.math.top_k(grads, k=N_BP_MESSAGES)

        # al = 0
        # siz = 0.1
        for i in range(N_BP_MESSAGES):
            # if the message has even index, its this agent's, otherwise its his teammates
            msg_agent_id = (id + (indexes[i] % 2) * 2) % 4
            # index of the message in self.messages array (and self.tapes)
            msg_index = indexes[i] // 2 + 1
            # step, when the message was sent
            msg_step = self.step - msg_index
            # index of the model that sent this message (in self.models)
            m_i = 0 if msg_step < 60 else 1 if msg_step < 180 else 2

            if self.tapes[msg_agent_id][msg_index] is None or sizes[i] < 1e-20:
                continue

            with self.tapes[msg_agent_id][msg_index]:
                # multiplied by an arbitrary number to combat vanishing gradient by upscaling everything to roughly
                # the same scale as the first model backprop
                msg = self.messages[msg_agent_id][msg_index] * chat_grads[:, :, :, indexes[i]] * 1e4

            m_g, c_m_g = self.tapes[msg_agent_id][msg_index].gradient(msg, [self.models[m_i].trainable_variables,
                                                                            self.chat_model.trainable_variables])

            self.save_grads(m_g, c_m_g, m_i, msg_agent_id, weight=0.5/N_BP_MESSAGES)
            # for i in range(len(m_g)):
            #     if m_g[i] is not None:
            #         al += np.sum(np.absolute(m_g[i]))
            #         siz += np.array(m_g[i]).size
        # print(al / siz)


    def training_step(self, agent_features, a_id, position):
        tim = time.time()
        new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        # index of the model to use
        m_i = 0 if self.step < 60 else 1 if self.step < 180 else 2

        chat = self.get_chat(a_id)
        a = time.time() - tim
        tim = time.time()
        with new_tape:
            new_tape.watch(self.models[m_i].trainable_variables)
            new_tape.watch(self.chat_model.trainable_variables)
            new_tape.watch(chat)
            b = time.time() - tim
            tim = time.time()

            chat_features = self.chat_model(chat)
            features = tf.concat([agent_features[:, :, :, :N_MAP_FEATURES], chat_features], 3)
            actions, msg = self.models[m_i](features)
            c = time.time() - tim
            self.add_message(msg, a_id)

            tim = time.time()
            action_filter = AF.get_action_filter(position[0], position[1], features)
            action = AF.apply_action_filter(action_filter, actions[0])
            d = time.time() - tim

            # if all actions result in a certain death, don't compute gradients and just return 0
            if action is None:
                self.tapes[a_id][0] = new_tape
                return 0
            tim = time.time()
            # computes loss in regards to the first possible choice of the model, that doesn't result in a certain death
            loss = self.compute_loss([action], actions[0])
        e = time.time() - tim
        tim = time.time()
        model_grads, chat_model_grads, chat_grads = new_tape.gradient(loss, [self.models[m_i].trainable_variables,
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
        self.backprop_chat(chat_grads, a_id)
        f = time.time() - tim
        tim = time.time()
        self.save_grads(model_grads, chat_model_grads, m_i, a_id, weight=0.5)

        self.tapes[a_id][0] = new_tape
        g = time.time() - tim
        #         print("all", round(a+b+c+d+e+f+g, 3), "get_vars", round(a, 3), "watch", round(b, 3), "forward", round(c, 3), "filter", round(d, 3), "losscomp+close", round(e, 3), "grads1", round(h, 3), "msggrads", round(f, 3), "addgrads", round(g, 3))

        return action
