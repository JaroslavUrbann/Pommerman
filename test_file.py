import tensorflow as tf
tf.enable_eager_execution()

w = tf.Variable(2.0)
tape = tf.GradientTape(watch_accessed_variables=True, persistent=True)
tape2 = tf.GradientTape(watch_accessed_variables=True, persistent=True)
with tape:
    tape.watch(w)
with tape2:
    tape2.watch(w)
tapes = [None, tape]


def nn(x):
    global w
    return w * x[0] + w + x[1]


def compute_loss(y):
    return y


def add_message(message):
    global msgs
    msgs = msgs[-1:] + msgs[:-1]
    msgs[0] = [message]


msgs = [tf.Variable(tf.zeros((1,))) for _ in range(2)]
loss = 0

from contextlib import ExitStack

with ExitStack() as stack:
    for t in tapes:
        stack.enter_context(t)
# with tape:

    y = nn(tf.concat(msgs, 0))
    loss1 = compute_loss(y)
    loss += loss1
    add_message(y)
    # print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(msgs, 0))
    loss2 = compute_loss(y)
    tf.stop_gradient(loss2)
    loss += loss2
    add_message(y)
    # print(tape.gradient(loss, w))
    del y

tapes.append(tape2)
with ExitStack() as stack:
    for t in tapes:
        stack.enter_context(t)

    y = nn(tf.concat(msgs, 0))
    loss3 = compute_loss(y)
    loss += loss3
    add_message(y)
    # print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(msgs, 0))
    loss4 = compute_loss(y)
    loss += loss4
    add_message(y)
    # print(tape.gradient(loss, w))
    del y
print(tapes[1].gradient(loss, w))