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
    global messages
    messages = messages[-1:] + messages[:-1]
    messages[0] = [message]


messages = [tf.Variable(tf.zeros((1,))) for _ in range(2)]
loss = 0


with tape:
    tape.watch(w)

    y = nn(tf.concat(messages, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(messages, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(messages, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y

    y = nn(tf.concat(messages, 0))
    loss += compute_loss(y)
    add_message(y)
    print(tape.gradient(loss, w))
    del y
