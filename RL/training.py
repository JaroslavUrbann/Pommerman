import tensorflow as tf
from constants import *
from RL import action_filter as AF
import time
import numpy as np


def reset_grads():
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    agents_grads[:] = 0
    chats_grads[:] = 0


# died_first should ideally have at most 1 player from each team
def apply_grads(won, died_first):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    # 0.7 if it won but died, -0.7 if lost but didn't die first, else -1/1
    rewards = [(0.7 if a in died_first else 1) if a in won else -1 if a in died_first else -0.7 for a in range(4)]

    agents_grads[0] *= rewards[0]
    agents_grads[1] *= rewards[1]
    agents_grads[2] *= rewards[2]
    agents_grads[3] *= rewards[3]

    a_grads = np.sum(agents_grads, axis=0) / 4

    chats_grads[0] *= rewards[0]
    chats_grads[1] *= rewards[1]
    chats_grads[2] *= rewards[2]
    chats_grads[3] *= rewards[3]

    c_grads = np.sum(chats_grads, axis=0) / 4

    optimizer.apply_gradients(zip(a_grads, model.trainable_variables))
    optimizer.apply_gradients(zip(c_grads, chat_model.trainable_variables))


def end_episode(won, died_first):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    if won:
        apply_grads(won, died_first)
    reset_grads()

    # this timestep + 32 previous timesteps
    n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
    tapes = [[None] * n_msgs for _ in range(4)]
    messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]


def end_step():
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    for i in range(4):
        tapes[i] = [None] + tapes[i][:-1]
        messages[i] = [tf.zeros((1, 3, 3))] + messages[i][:-1]


def get_chat(id):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    chat = [None] * CHAT_HISTORY_LENGTH
    chat[0::2] = messages[id][1:]
    chat[1::2] = messages[(id + 2) % 4][1:]
    chat = tf.stack(chat, 3)
    return chat


def add_message(msg, id):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    # add noise and change to 0 - 1 distribution
    msg += tf.random.normal(msg.shape, mean=0.0, stddev=10.0)
    msg = tf.math.sigmoid(msg)

    # reshape msg and add it to stack
    msg = tf.reshape(msg, (1, 2, 3))
    padding = tf.zeros((1, 1, 3))
    msg = tf.concat([msg, padding], 1)
    messages[id][0] = msg


# adds gradients from pervious timesteps to gradients from this timestep
# how much weight is given to gradients from previous timesteps depends on GRADIENT_DISCOUNT
def add_grads(agent_grads, chat_grads, id, step):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    # equation for first n elements of a geometric sequence
    total_avg = (1 - GRADIENT_DISCOUNT ** (step + 1)) / (1 - GRADIENT_DISCOUNT)

    agent_grads[np.isnan(agent_grads)] = 0
    agents_grads[id] = (agents_grads[id] * (total_avg - 1) + agent_grads) / total_avg

    # are later chats really more important? questionable decision
    chat_grads[np.isnan(chat_grads)] = 0
    chats_grads[id] = (chats_grads[id] * (total_avg - 1) + chat_grads) / total_avg


def backprop_chat(chat_grads, model_grads, chat_model_grads, id):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    abs_grads = tf.abs(chat_grads)
    grads = tf.reduce_sum(abs_grads, [0, 1, 2])
    sizes, indexes = tf.math.top_k(grads, k=N_BP_MESSAGES)

#     al = 0
#     siz = 0.1
    for i in range(N_BP_MESSAGES):
        # if the message has even index, its this agent's, otherwise its his teammates
        msg_agent_id = (id + (indexes[i] % 2) * 2) % 4
        msg_id = indexes[i] // 2 + 1

        # if the message is just a default 0 message (it doesn't have a tape to it)
        # or if the gradient is too low, just continue
        if tapes[msg_agent_id][msg_id] is None or sizes[i] < 1e-20:
            continue

        with tapes[msg_agent_id][msg_id]:
            msg = messages[msg_agent_id][msg_id] * chat_grads[:, :, :, indexes[i]] * 1e3 # arbitrary number to combat vanishing gradient by upscaling everything to roughly the same scale as the first model backprop
        m_g, c_m_g = tapes[msg_agent_id][msg_id].gradient(msg, [model.trainable_variables,
                                                                 chat_model.trainable_variables])
#         for i in range(len(m_g)):
#             if m_g[i] is not None:
#                 al += np.sum(np.absolute(m_g[i]))
#                 siz += np.array(m_g[i]).size

        model_grads += m_g
        chat_model_grads += c_m_g
#     print(al/siz)
    return model_grads, chat_model_grads


def training_step(agent_features, id, step, position):
    tim = time.time()
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)

    chat = get_chat(id)
    a = time.time() - tim
    tim = time.time()
    with new_tape:
        new_tape.watch(model.trainable_variables)
        new_tape.watch(chat_model.trainable_variables)
        new_tape.watch(chat)
        b = time.time() - tim
        tim = time.time()

        chat_features = chat_model(chat)
        features = tf.concat([agent_features[:, :, :, :N_MAP_FEATURES], chat_features], 3)
        actions, msg = model(features)
        c = time.time() - tim
        tim = time.time()
        add_message(msg, id)

        action_filter = AF.get_action_filter(position[0], position[1], features)
        action = AF.apply_action_filter(action_filter, actions[0])

        # if all actions result in a certain death, don't compute gradients and just return 0
        if action is None:
            tapes[id][0] = new_tape
            return 0
        d = time.time() - tim
        tim = time.time()
        # computes loss in regards to the first possible choice of the model, that doesn't result in a certain death
        loss = compute_loss([action], actions[0])
    e = time.time() - tim
    tim = time.time()
    model_grads, chat_model_grads, chat_grads = new_tape.gradient(loss, [model.trainable_variables,
                                                                         chat_model.trainable_variables, chat])
#     print(time.time() - tim, "------------------")
#     al = 0
#     siz = 0
#     for i in range(len(model_grads)):
#         if model_grads[i] is not None:
#             al += np.sum(np.absolute(model_grads[i]))
#             siz += np.array(model_grads[i]).size
#     print(al/siz, "-------------------")

#     tim = time.time()
    model_grads, chat_model_grads = backprop_chat(chat_grads, model_grads, chat_model_grads, id)
    f = time.time() - tim
    tim = time.time()
    add_grads(model_grads.numpy(), chat_model_grads.numpy(), id, step)

    tapes[id][0] = new_tape
    g = time.time() - tim
    print("all", round(a+b+c+d+e+f+g, 3), "get_vars", round(a, 3), "watch", round(b, 3), "forward", round(c, 3), "addmsg+filter", round(d, 3), "losscomp+close", round(e, 3), "grads", round(f, 3), "addgrads", round(g, 3))

    return action


def init_training(m, c_m):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    model = m
    chat_model = c_m

    agents_grads = np.array([model.trainable_variables,
                             model.trainable_variables,
                             model.trainable_variables,
                             model.trainable_variables], dtype="float32")
    chats_grads = np.array([chat_model.trainable_variables,
                            chat_model.trainable_variables,
                            chat_model.trainable_variables,
                            chat_model.trainable_variables], dtype="float32")

    # this timestep + 32 previous timesteps
    n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
    tapes = [[None] * n_msgs for _ in range(4)]
    messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=RL_LR)

    reset_grads()