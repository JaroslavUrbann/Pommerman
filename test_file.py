import tensorflow as tf
tf.enable_eager_execution()

w = tf.Variable(2.0)
tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)


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


with tape:
    tape.watch(w)

    y = nn(tf.concat(msgs, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(msgs, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(msgs, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(msgs, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y
