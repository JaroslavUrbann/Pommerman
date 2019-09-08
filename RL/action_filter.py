import tensorflow as tf


# returns the first possible action chosen by the model, that doesn't result in a certain death,
# None, if all actions result in a certain death
def apply_action_filter(action_filter, probabilities):
    _, indices = tf.math.top_k(probabilities)
    i = 0
    while action_filter[indices[i]] == 0:
        i += 1
        if i == len(indices):
            return None
    return probabilities[indices[i]]


def get_action_filter(position, features):
    a0 = is_square_safe(position[0], position[1], features)
    a1 = is_square_safe(position[0] - 1, position[1], features)
    a2 = is_square_safe(position[0], position[1] - 1, features)
    a3 = is_square_safe(position[0] + 1, position[1], features)
    a4 = is_square_safe(position[0], position[1] + 1, features)
    a5 = 0 if features[0, position[0], position[1], 9] > 0 else 1
    return [a0, a1, a2, a3, a4, a5]


def is_square_lethal(row, col, features):
    # is it outside the map?
    if col < 0 or col > 10 or row < 0 or row > 10:
        return True
    # is there wood or stone?
    if features[0, row, col, 0] == 1 or features[0, row, col, 1] == 1:
        return True
    # will there be a flame next timestep? (from a previous flame)
    if features[0, row, col, 11] > 0.1:
        return True
    # is there a bomb that explodes next timestep?
    if features[0, row, col, 9] == 0.9:
        return True


# currently, I will be using is_square_safe() only to determine if an action leads to certain death,
# so if an agent chooses an action with a low survival probability, I will let him.
# I am leaving this code here to perhaps implement a safe agent in the future,
# that always chooses an action with survival probability of 1
def is_square_safe(row, col, features):
    # absolutely no chance of stepping on or surviving on this square section:
    
    if is_square_lethal(row, col, features):
        return 0

    # # is there a bomb and how safe is the space behind it?
    # # (cat it be moved, is there a possibility that there will be flames next timestep etc.)
    # if features[0, row, col, 9] > 0:
    #     safeness_of_space_behind_bomb = 0.1
    #     if row < 10 and features[0, row + 1, col, 7] == 1:
    #         return is_square_safe(row - 1, col, features)
    #     if col < 10 and features[0, row, col + 1, 7] == 1:
    #         return is_square_safe(row, col - 1, features)
    #     if row > 0 and features[0, row - 1, col, 7] == 1:
    #         return is_square_safe(row + 1, col, features)
    #     if col > 0 and features[0, row, col - 1, 7] == 1:
    #         return is_square_safe(row, col + 1, features)
    #     else:  # there are 2 bombs after each other and the second one could be moving / start moving
    #         return ?
    #
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
