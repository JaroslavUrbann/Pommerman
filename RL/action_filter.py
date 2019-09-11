import tensorflow as tf
from constants import *


# check a square is surrounded by unmovable blocks
# created to prevent agents from going into 1x1 traps after placing a bomb, that they can't kick
def is_square_a_trap(row, col, features):
    if is_square_blocked(row, col, features, (0, -1)) and \
        is_square_blocked(row, col, features, (+1, 0)) and \
        is_square_blocked(row, col, features, (0, +1)) and \
        is_square_blocked(row, col, features, (-1, 0)):
        return True
    return False


# checks if a square is blocked (needs direction to check if you can move a bomb)
def is_square_blocked(row, col, features, direction):
    row += direction[0]
    col += direction[1]
    # is it outside the map?
    if col < 0 or col > 10 or row < 0 or row > 10:
        return True
    # is there wood or stone?
    if features[0, row, col, 0] == 1 or features[0, row, col, 1] == 1:
        return True
    # is there a bomb and the agent can't kick or is there a bomb and the space behind it (according to direction) blocked?
    # if there are 2 bombs after each other, this breaks, since you can't push two bombs at the same time
    if features[0, row, col, 9] > 0 and (features[0, row, col, 30] == 0 or is_square_blocked(row, col, features, direction)):
        return True
    return False


# returns the first possible action chosen by the model, that doesn't result in a certain death,
# None, if all actions result in a certain death
def apply_action_filter(action_filter, probabilities):
    _, indices = tf.math.top_k(probabilities, k=N_ACTIONS)
    i = 0
    while action_filter[indices[i]] == 0:
        i += 1
        if i == len(indices):
            return None
    return indices[i]


def get_action_filter(row, col, features):
    a0 = is_square_safe(row, col, features, (0, 0))
    a1 = is_square_safe(row, col, features, (-1, 0))
    a2 = is_square_safe(row, col, features, (+1, 0))
    a3 = is_square_safe(row, col, features, (0, -1))
    a4 = is_square_safe(row, col, features, (0, +1))
    a5 = 0 if features[0, row, col, 9] > 0 else 1
    # stop, up, down, left, right, place_bomb
    return [a0, a1, a2, a3, a4, a5]


# currently, I will be using is_square_safe() only to determine if an action leads to certain death,
# so if an agent chooses an action with a low survival probability, I will let him.
# I am leaving this code here to perhaps implement a safe agent in the future,
# that always chooses an action with survival probability of 1
def is_square_safe(row_, col_, features, direction):
    row = row_ + direction[0]
    col = col_ + direction[1]
    # absolutely no chance of stepping on or surviving on this square section:
    # is it outside the map?
    if col < 0 or col > 10 or row < 0 or row > 10:
        return 0
    # is there wood or stone?
    if features[0, row, col, 0] == 1 or features[0, row, col, 1] == 1:
        return 0
    # will there be a flame next timestep? (from a previous flame)
    if features[0, row, col, 12] == 1:
        return 0
    # is there a bomb that explodes next timestep?
    if features[0, row, col, 9] > 0.8:
        return 0
    # is there a bomb under the agent, that explodes in t + 2?
    # doesnt't take into account chaining explosions!!
    if features[0, row, col, 9] > 0.7 and features[0, row, col, 7] == 1:
        return 0

    # is there a bomb and how safe is the space behind it?
    # (cat it be moved, is there a possibility that there will be flames next timestep etc.)
    if features[0, row, col, 9] > 0:
        # the agent is standing on top of the bomb (it is safe to stay there)
        if features[0, row, col, 7] == 1:
            return 1
        # if the agent can't kick -> not safe (can't go there)
        if features[0, row, col, 30] == 0:
            return 0
        # if the agent is standing to the side of the bomb, can he push it? (is the square behind it safe?)
        # if there are 2 bombs after each other, this breaks, since you can't push two bombs at the same time
        return is_square_safe(row, col, features, direction)

    # is square a trap?
    if features[0, row_, col_, 9] > 0 and features[0, row_, col_, 7] == 1 and is_square_a_trap(row, col, features):
        return 0

    # this will need reworking if I ever use it, since I changed the flame / blast strength system
    # # uncertainty section:
    # # calculating the exact position of a bomb / enemy / teammate in the next step is either very hard or impossible
    # # (the bomb can be moving / start moving at any time in any direction etc.)
    # # therefore, the next section returns hand-crafted safety scores
    #
    # # is a bomb expected to explode there in the next timestep,
    # # that could technically be moving / start moving and not end up exploding there?
    # if features[0, row, col, 12] == 0.9:
    #     return ?
    #
    # # is an unexploded bomb nearby, that could be moving / start moving and explode there?
    # if (row < 10 and features[0, row + 1, col, 12] == 0.9) or (row > 0 and features[0, row - 1, col, 12] == 0.9) or \
    #     (col < 10 and features[0, row, col + 1, 12] == 0.9) or (col > 0 and features[0, row, col - 1, 12] == 0.9):
    #     return ?
    #
    # # is there a player (enemy / teammate), that could move away from this square?
    # if features[0, row, col, 5] == 1 or features[0, row, col, 6] == 1:
    #     return ?
    #
    # # is a player (enemy / teammate) nearby, that could move into that position?
    # if (row < 10 and features[0, row + 1, col, 5] == 0.9) or (row > 0 and features[0, row - 1, col, 5] == 0.9) or \
    #     (col < 10 and features[0, row, col + 1, 5] == 0.9) or (col > 0 and features[0, row, col - 1, 5] == 0.9) or \
    #     (row < 10 and features[0, row + 1, col, 6] == 0.9) or (row > 0 and features[0, row - 1, col, 6] == 0.9) or \
    #     (col < 10 and features[0, row, col + 1, 6] == 0.9) or (col > 0 and features[0, row, col - 1, 6] == 0.9):
    #     return ?
    return 1