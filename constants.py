BOARD_SIZE = (11, 11)
WOOD = 2
STONE = 1
AMMO_POWERUP = 6
RANGE_POWERUP = 7
KICK_POWERUP = 8
FOG = 5
TEAMMATE = "teammate"
ENEMIES = "enemies"
AGENT = "agent"
PLAYER_DECAY = 0.5
BOMB = 3

N_CLASSES = 6
LR = 3e-4
RL_LR = 1e-7
N_FEATURES = 24
N_MESSAGE_BITS = 6
CHAT_HISTORY_LENGTH = 64  # number of messages from each agent * 2
GRADIENT_DISCOUNT = 0.999
N_BP_MESSAGES = 32