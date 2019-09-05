import tensorflow as tf
from constants import *
import time
import numpy as np


def reset_grads():
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    for i in range(len(agents_grads[0])):
        agents_grads[0][i] = agents_grads[0][i] * 0
        agents_grads[1][i] = agents_grads[1][i] * 0
        agents_grads[2][i] = agents_grads[2][i] * 0
        agents_grads[3][i] = agents_grads[3][i] * 0
    for i in range(len(chats_grads[0])):
        chats_grads[0][i] = chats_grads[0][i] * 0
        chats_grads[1][i] = chats_grads[1][i] * 0
        chats_grads[2][i] = chats_grads[2][i] * 0
        chats_grads[3][i] = chats_grads[3][i] * 0


# died_first should ideally have at most 1 player from each team
def apply_grads(won, died_first):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    # 0.7 if it won but died, -0.7 if lost but didn't die first, else -1/1
    rewards = [(0.7 if a in died_first else 1) if a in won else -1 if a in died_first else -0.7 for a in range(4)]

    for var in range(len(agents_grads[0])):
        grads_agent_0 = agents_grads[0][var] * rewards[0]
        grads_agent_1 = agents_grads[1][var] * rewards[1]
        grads_agent_2 = agents_grads[2][var] * rewards[2]
        grads_agent_3 = agents_grads[3][var] * rewards[3]
        agents_grads[0][var] = (grads_agent_0 + grads_agent_1 + grads_agent_2 + grads_agent_3) / 4

    for var in range(len(chats_grads[0])):
        grads_msgs_0 = chats_grads[0][var] * rewards[0]
        grads_msgs_1 = chats_grads[1][var] * rewards[1]
        grads_msgs_2 = chats_grads[2][var] * rewards[2]
        grads_msgs_3 = chats_grads[3][var] * rewards[3]
        chats_grads[0][var] = (grads_msgs_0 + grads_msgs_1 + grads_msgs_2 + grads_msgs_3) / 4

    optimizer.apply_gradients(zip(agents_grads[0], model.trainable_variables))
    optimizer.apply_gradients(zip(chats_grads[0], chat_model.trainable_variables))


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

    for a in range(len(agents_grads[0])):
        # grads will be None at message layers
        if agent_grads[a] is None:
            continue
        agents_grads[id][a] = (agents_grads[id][a] * (total_avg - 1) + agent_grads[a]) / total_avg

    # are later chats really more important? questionable decision
    for m in range(len(chats_grads[0])):
        chats_grads[id][m] = (chats_grads[id][m] * (total_avg - 1) + chat_grads[m]) / total_avg


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

        if tapes[msg_agent_id][msg_id] is None or sizes[i] < 1e-20:
            continue

        with tapes[msg_agent_id][msg_id]:
            msg = messages[msg_agent_id][msg_id] * chat_grads[:, :, :, indexes[i]] * 1e4 # arbitrary number to combat vanishing gradient by upscaling everything to roughly the same scale as the first model backprop
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


def training_step(agent_features, id, step):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    new_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)

    chat = get_chat(id)

    with new_tape:
        new_tape.watch(model.trainable_variables)
        new_tape.watch(chat_model.trainable_variables)
        new_tape.watch(chat)

        chat_features = chat_model(chat)
        features = tf.concat([agent_features[:, :, :, :21], chat_features], 3)
        actions, msg = model(features)

        add_message(msg, id)

        action = tf.math.argmax(actions[0])
        loss = compute_loss([action], actions[0])

    a = time.time()
    model_grads, chat_model_grads, chat_grads = new_tape.gradient(loss, [model.trainable_variables,
                                                                         chat_model.trainable_variables, chat])
#     al = 0
#     siz = 0
#     for i in range(len(model_grads)):
#         if model_grads[i] is not None:
#             al += np.sum(np.absolute(model_grads[i]))
#             siz += np.array(model_grads[i]).size
#     print(al/siz, "-------------------")
    b = time.time() - a
    a = time.time()
    model_grads, chat_model_grads = backprop_chat(chat_grads, model_grads, chat_model_grads, id)
    c = time.time() - a
    add_grads(model_grads, chat_model_grads, id, step)

    tapes[id][0] = new_tape
#     print(c)
    return action


def init_training(m, c_m):
    global model, chat_model, agents_grads, chats_grads, tapes, messages, compute_loss, optimizer
    model = m
    chat_model = c_m

    agents_grads = [model.trainable_variables] * 4
    chats_grads = [chat_model.trainable_variables] * 4

    # this timestep + 32 previous timesteps
    n_msgs = int((1 + CHAT_HISTORY_LENGTH / 2))
    tapes = [[None] * n_msgs for _ in range(4)]
    messages = [[tf.zeros((1, 3, 3))] * n_msgs for _ in range(4)]

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=RL_LR)

    reset_grads()