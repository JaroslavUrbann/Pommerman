import tensorflow as tf
from constants import *
from contextlib import ExitStack
import time

tf.enable_eager_execution()


@tf.custom_gradient
def regulate_grad(x):
    def _grad(dy):
        abs_grads = tf.abs(dy)
        grads = tf.reduce_sum(abs_grads, [0, 1, 2])
        _, indexes = tf.math.top_k(grads, k=N_BP_MESSAGES)

        avg_all = tf.reduce_sum(grads, axis=0) / CHAT_HISTORY_LENGTH / 6
        avg_biggest = tf.reduce_sum(_, axis=0) / N_BP_MESSAGES / 6
        print(avg_all, avg_biggest)

        new_grads = tf.zeros_like(dy)
        for i in range(N_BP_MESSAGES):
            new_grads[:, :, :, indexes[i]].assign(dy[:, :, :, indexes[i]] * 0.5 / N_BP_MESSAGES)
        return new_grads

    return x, _grad


class RLTraining:

    def __init__(self, model, chat_model):
        self.model = model
        self.chat_model = chat_model
        self.agents_grads = [model.trainable_variables for _ in range(4)]
        self.chats_grads = [chat_model.trainable_variables for _ in range(4)]
        tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        with tape:
            tape.watch(model.trainable_variables)
            tape.watch(chat_model.trainable_variables)
        self.tapes = [tape]
        self.chats = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(CHAT_HISTORY_LENGTH)] for _ in range(4)]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
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

    def add_grads(self, agent_grads, chat_grads, id, step):
        # equation for first n elements of a geometric sequence
        total_discount = (1 - GRADIENT_DISCOUNT ** (step + 1)) / (1 - GRADIENT_DISCOUNT)
        for a in range(len(self.agents_grads[0])):
            if agent_grads[a] is None:
                continue
            new_avg = (self.agents_grads[id][a] * (total_discount - 1) + agent_grads[a]) / total_discount
            self.agents_grads[id][a] = new_avg

        # are later chats really more important? questionable decision
        for m in range(len(self.chats_grads[0])):
            new_avg = (self.chats_grads[id][m] * (total_discount - 1) + chat_grads[m]) / total_discount
            self.chats_grads[id][m] = new_avg

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
        self.optimizer.apply_gradients(zip(self.agents_grads[0], self.model.trainable_variables))

        for var in range(len(self.chats_grads[0])):
            grads_msgs_0 = self.chats_grads[0][var] * rewards[0]
            grads_msgs_1 = self.chats_grads[1][var] * rewards[1]
            grads_msgs_2 = self.chats_grads[2][var] * rewards[2]
            grads_msgs_3 = self.chats_grads[3][var] * rewards[3]
            self.chats_grads[0][var] = (grads_msgs_0 + grads_msgs_1 + grads_msgs_2 + grads_msgs_3) / 4
        self.optimizer.apply_gradients(zip(self.chats_grads[0], self.chat_model.trainable_variables))

    def end_episode(self, won, died_first):
        if won:
            self.apply_grads(won, died_first)
        self.reset_grads()
        self.chats = [[tf.Variable(tf.zeros((1, 3, 3, 1))) for _ in range(CHAT_HISTORY_LENGTH)] for _ in range(2)]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]
        new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        with new_tape:
            new_tape.watch(self.model.trainable_variables)
            new_tape.watch(self.chat_model.trainable_variables)
        self.tapes.append(new_tape)
        if len(self.tapes) > N_TAPES:
            del self.tapes[0]

    # exchanges messages, is needed because some agents might be dead and can't add theirs to the chat
    def next_time_step(self):
        for c in range(4):
            self.chats[c] = self.chats[c][-2:] + self.chats[c][:-2]
            self.chats[c][0] = self.next_msgs[c]
            self.chats[c][1] = self.next_msgs[(c + 2) % 4]
        self.next_msgs = [tf.zeros((1, 3, 3, 1)) for _ in range(4)]

    def training_step(self, agent_features, id, step):
        with ExitStack() as stack:
            for t in self.tapes:
                stack.enter_context(t)

            tim = time.time()
            # takes chat conversation
            chat = tf.concat(self.chats[id], 3)
            chat = regulate_grad(chat)

            # gets chat features to put in his network
            chat_features = self.chat_model(chat)

            features = tf.concat([agent_features[:, :, :, :21], chat_features], 3)
            a = time.time() - tim

            tim = time.time()
            actions, msg = self.model(features)
            b = time.time() - tim

            tim = time.time()
            # add noise and change to 0 - 1 distribution
            msg += tf.random.normal(msg.shape, mean=0.0, stddev=10.0)
            msg = tf.math.sigmoid(msg)

            # reshape msg and add it to stack
            msg = tf.reshape(msg, (1, 2, 3, 1))
            padding = tf.zeros((1, 1, 3, 1))
            msg = tf.concat([msg, padding], 1)
            self.next_msgs[id] = msg

            action = tf.math.argmax(actions[0])
            # / 2 is so that half the gradients for this timestep come from current action
            loss = self.compute_loss([action], actions[0]) / 2
            c = time.time() - tim

        tim = time.time()
        agent_grads, chat_grads = self.tapes[0].gradient(loss, [self.model.trainable_variables, self.chat_model.trainable_variables])
        d = time.time() - tim

        tim = time.time()
        self.add_grads(agent_grads, chat_grads, id, step)
        e = time.time() - tim
        print("features + message prediction: " + str(a) + " model prediction: " + str(
            b) + " msg shaping + loss predicting: " + str(c) + " gradient calculating: " + str(
            d) + " gradient adding: " + str(e))

        return action
