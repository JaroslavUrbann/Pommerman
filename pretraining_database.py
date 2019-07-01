import numpy as np
import random


MAX_STEPS = 36000
step_index = 0
x_agent_1 = [{} for _ in range(MAX_STEPS)]
x_agent_2 = [{} for _ in range(MAX_STEPS)]
y_agent_1 = [0 for _ in range(MAX_STEPS)]
y_agent_2 = [0 for _ in range(MAX_STEPS)]


def reset_database():
    global x_agent_1, x_agent_2, y_agent_1, y_agent_2, step_index
    step_index = 0
    x_agent_1 = [{} for _ in range(MAX_STEPS)]
    x_agent_2 = [{} for _ in range(MAX_STEPS)]
    y_agent_1 = [0 for _ in range(MAX_STEPS)]
    y_agent_2 = [0 for _ in range(MAX_STEPS)]


def shuffle_database():
    global x_agent_1, x_agent_2, y_agent_1, y_agent_2
    c = list(zip(x_agent_1, x_agent_2, y_agent_1, y_agent_2))
    random.shuffle(c)
    x_agent_1, x_agent_2, y_agent_1, y_agent_2 = zip(*c)
