import tensorflow as tf
from constants import *


def training(model, message_model):
    model = model
    message_model = message_model
    messages = [tf.zeros((3, 3, MESSAGE_HISTORY_LENGTH)) for _ in range(4)]
