import numpy as np
import time
from constants import *


class FeatureEngineer:

    def __init__(self, CN=None):
        self.CN = CN
        self.messages = [np.zeros((1, 3, 3))] * CHAT_HISTORY_LENGTH
        self._chat_features_map = np.zeros(BOARD_SIZE + (4,))

        self._wood_map = np.zeros(BOARD_SIZE)
        self._stone_map = np.zeros(BOARD_SIZE)
        self._ammo_powerup_map = np.zeros(BOARD_SIZE)
        self._range_powerup_map = np.zeros(BOARD_SIZE)
        self._kick_powerup_map = np.zeros(BOARD_SIZE)
        self._fog_map = np.zeros(BOARD_SIZE)

        self._agent_map = np.zeros(BOARD_SIZE)
        self._teammate_map = np.zeros(BOARD_SIZE)
        self._enemies_map = np.zeros(BOARD_SIZE)
        self._agent_history_map = np.zeros(BOARD_SIZE)
        self._teammate_history_map = np.zeros(BOARD_SIZE)
        self._enemies_history_map = np.zeros(BOARD_SIZE)

        self._bomb_map = np.zeros(BOARD_SIZE)
        self._bomb_history_map = np.zeros(BOARD_SIZE) # used to possibly show moving direction of bombs
        self._hidden_blast_strength_map = np.zeros(BOARD_SIZE)
        self._blast_strength_map = np.zeros(BOARD_SIZE)
        self._blast_strength_map_1 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_2 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_3 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_4 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_5 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_6 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_7 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_8 = np.zeros(BOARD_SIZE)
        self._blast_strength_map_9 = np.zeros(BOARD_SIZE)
        self._flame_map_1 = np.zeros(BOARD_SIZE)
        self._flame_map_2 = np.zeros(BOARD_SIZE)
        self._flame_map_3 = np.zeros(BOARD_SIZE)

        self._ammo1_map = np.zeros(BOARD_SIZE)
        self._ammo2_map = np.zeros(BOARD_SIZE)
        self._ammo3_map = np.zeros(BOARD_SIZE)
        self._ammo4_map = np.zeros(BOARD_SIZE)
        self._blast1_map = np.zeros(BOARD_SIZE)
        self._blast2_map = np.zeros(BOARD_SIZE)
        self._kick_map = np.zeros(BOARD_SIZE)

        self._features = np.zeros((1, BOARD_SIZE[0], BOARD_SIZE[1], N_FEATURES), dtype="float32")

    # _agent_number = 0
    #
    # def _get_agent_number(self, observation):
    #     if not self._agent_number:
    #         self._agent_number = 10 + 11 + 12 + 13 - observation[TEAMMATE].value - observation[ENEMIES][0].value - observation[ENEMIES][1].value
    #     return self._agent_number


    # gets message from game format [1,8] to binary format in shape (1, 3, 3) (using only first 2 rows)
    def _dec_to_binary(self, message):
        msg = [0, 0]
        msg[0] = max(0, message[0] - 1)
        msg[1] = max(0, message[1] - 1)
        dec = msg[0] * 8 + msg[1]
        binary = format(dec, '06b')
        msg = np.zeros((1, 3, 3))
        msg[0, 0, 0] = int(binary[0])
        msg[0, 0, 1] = int(binary[1])
        msg[0, 0, 2] = int(binary[2])
        msg[0, 1, 0] = int(binary[3])
        msg[0, 1, 1] = int(binary[4])
        msg[0, 1, 2] = int(binary[5])
        return msg


    # (my message, teammates mesage) from previous round
    def _update_chat_features_map(self, messages):
        self.messages = [self._dec_to_binary(messages[0]), self._dec_to_binary(messages[1])] + self.messages[2:]
        chat = np.stack(self.messages, 3)
        chat = np.array(chat, dtype="float32")
        self._chat_features_map = self.CN(chat).numpy()


    def _get_fov_boundries(self, observation):
        # player position
        row = observation["position"][0]
        col = observation["position"][1]
        # players field of view (clipped)
        top = max(0, row - 4)
        bottom = min(row + 5, BOARD_SIZE[0])
        left = max(0, col - 4)
        right = min(col + 5, BOARD_SIZE[1])
        return top, bottom, left, right


    def _update_status_maps(self, observation):
        # ammo can be probably infinite, blast starts at 2 and goes to max 4
        self._ammo1_map = np.ones(BOARD_SIZE) if observation["ammo"] > 0 else np.zeros(BOARD_SIZE)
        self._ammo2_map = np.ones(BOARD_SIZE) if observation["ammo"] > 1 else np.zeros(BOARD_SIZE)
        self._ammo3_map = np.ones(BOARD_SIZE) if observation["ammo"] > 2 else np.zeros(BOARD_SIZE)
        self._ammo4_map = np.ones(BOARD_SIZE) if observation["ammo"] > 3 else np.zeros(BOARD_SIZE)
        self._blast1_map = np.ones(BOARD_SIZE) if observation["blast_strength"] > 2 else np.zeros(BOARD_SIZE)
        self._blast2_map = np.ones(BOARD_SIZE) if observation["blast_strength"] > 3 else np.zeros(BOARD_SIZE)
        self._kick_map = np.ones(BOARD_SIZE) if int(observation["can_kick"]) else np.zeros(BOARD_SIZE)


    # should be used for WOOD, STONE, POWERUPS
    def _update_materials_map(self, observation, map, material):

        # remove wood if there are flames
        if material == WOOD:
            map[self._flame_map_1 == 1] = 0

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["board"][top:bottom, left:right]

        # maps the squares in fov into 1/0 depending on the chosen material
        filtered_fov = np.where(fov == material, 1, 0)

        # rewrites visible part of the map with updated mappings
        map[top:bottom, left:right] = filtered_fov


    # has be updated before blast strength map, because blast strenght map depends on it
    def _update_bomb_map(self, observation):
        self._bomb_history_map = self._bomb_map.copy()

        # bomb map goes from 0.1 to 0.9 (1 is explosion and is not shown on this map)
        self._bomb_map[self._bomb_map > 0] += 0.1
        self._bomb_map[self._bomb_map > 0.9] = 0

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["bomb_life"][top:bottom, left:right]

        # maps the squares with a bomb to the bombs' life
        filtered_fov = np.where(fov > 0, 1 - fov / 10, 0)

        # rewrites visible part of the map with updated mappings
        self._bomb_map[top:bottom, left:right] = filtered_fov

        # updates blast strength
        self._hidden_blast_strength_map[top:bottom, left:right] = observation["bomb_blast_strength"][top:bottom,
                                                                  left:right]


    # has to be updated before blast strength map because it needs to work with blast strength map at T-1
    def _update_flame_map(self, observation):
        # flame map is a continuation of blast strength map
        # flame is first on maps 1,2,3 and then 1,2 and then 1

        self._flame_map_1 = np.zeros(BOARD_SIZE)
        self._flame_map_1[self._flame_map_2 == 1] = 1

        self._flame_map_2 = np.zeros(BOARD_SIZE)
        self._flame_map_2[self._flame_map_3 == 1] = 1

        self._flame_map_3 = np.zeros(BOARD_SIZE)

        self._flame_map_1[self._blast_strength_map == 0.9] = 1
        self._flame_map_2[self._blast_strength_map == 0.9] = 1
        self._flame_map_3[self._blast_strength_map == 0.9] = 1

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["flame_life"][top:bottom, left:right]

        # rewrites visible part of the map with updated mappings
        self._flame_map_1[top:bottom, left:right] = np.where(fov > 0, 1, 0)
        self._flame_map_2[top:bottom, left:right] = np.where(fov > 1, 1, 0)
        self._flame_map_3[top:bottom, left:right] = np.where(fov > 2, 1, 0)


    def _update_blast_strength_maps(self):
        # blast strength map always starts with a clean slate and just maps the current bomb map
        # to the expected explosion radius based on a hidden map called _hidden_blast_strength_map
        # (it calculates chained explosions etc, and the squares' values are a count up from 0.1 to 0.9)
        # (1 represents the explosion and is not shown on this map)

        # clear explosions that happened unexpectedly
        self._blast_strength_map = np.zeros(BOARD_SIZE)

        row, col = np.where(self._bomb_map > 0)

        # [row, col, blast strength, life]
        bombs = []
        for i in range(len(col)):
            bombs.append([row[i], col[i], int(self._hidden_blast_strength_map[row[i], col[i]] - 1),
                          self._bomb_map[row[i], col[i]]])

        # sorts bombs by bomb life
        # (biggest value first, because those bombs explode first and might be chained to other bombs,
        # whose life then I have to adjust)
        bombs.sort(key=lambda tup: tup[3], reverse=True)

        # creates expected blast radius & time for all bombs (including chaining explosions etc)
        while bombs:
            bomb_range = bombs[0][2]
            bomb_life = bombs[0][3]
            self._blast_strength_map[bombs[0][0], bombs[0][1]] = bomb_life

            # writes bomb life to _blast_strength_map
            # and if there is a bomb on this square, it changes its' bomb life in the bombs array
            def _check4bomb(row, col, bomb_life):
                # two bombs could be expected to explode at one square
                # and so I want to show the earliest possibility of flames (therefore the bigger number)
                if self._blast_strength_map[row, col] < bomb_life:
                    self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    i = [i for i in range(len(bombs)) if bombs[i][0] == row and bombs[i][1] == col]
                    if i:
                        bombs[i[0]][3] = bomb_life

            # while in the map and in range and not on a stone square and while the previous square wasn't wood
            # right
            row, col = bombs[0][0], bombs[0][1] + 1
            a = 0
            while col < BOARD_SIZE[1] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[
                row, col - 1] != 1:
                _check4bomb(row, col, bomb_life)
                col += 1
                a += 1

            # left
            row, col = bombs[0][0], bombs[0][1] - 1
            a = 0
            while col >= 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row, col + 1] != 1:
                _check4bomb(row, col, bomb_life)
                col -= 1
                a += 1

            # down
            row, col = bombs[0][0] + 1, bombs[0][1]
            a = 0
            while row < BOARD_SIZE[0] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[
                row - 1, col] != 1:
                _check4bomb(row, col, bomb_life)
                row += 1
                a += 1

            # up
            row, col = bombs[0][0] - 1, bombs[0][1]
            a = 0
            while row >= 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row + 1, col] != 1:
                _check4bomb(row, col, bomb_life)
                row -= 1
                a += 1

            bombs.pop(0)

        # code up there in this function is a clusterfuck, so I will just ignore it
        # and build the new bomb blast strength system based on the map that it creates:

        self._blast_strength_map_1 = np.where(self._blast_strength_map > 0, 1, 0)
        self._blast_strength_map_2 = np.where(self._blast_strength_map > 0.1, 1, 0)
        self._blast_strength_map_3 = np.where(self._blast_strength_map > 0.2, 1, 0)
        self._blast_strength_map_4 = np.where(self._blast_strength_map > 0.3, 1, 0)
        self._blast_strength_map_5 = np.where(self._blast_strength_map > 0.4, 1, 0)
        self._blast_strength_map_6 = np.where(self._blast_strength_map > 0.5, 1, 0)
        self._blast_strength_map_7 = np.where(self._blast_strength_map > 0.6, 1, 0)
        self._blast_strength_map_8 = np.where(self._blast_strength_map > 0.7, 1, 0)
        self._blast_strength_map_9 = np.where(self._blast_strength_map > 0.8, 1, 0)


    def _update_players_map(self, observation):
        # player maps are maps where the known position of a player is represented by "1"
        # historical positions are represented with 1 to 0.1 (1 is current position, 0.1 is ten steps back)

        # current positions
        self._agent_map = np.zeros(BOARD_SIZE)
        self._agent_map[observation["position"][0], observation["position"][1]] = 1

        self._teammate_map = np.zeros(BOARD_SIZE)
        self._teammate_map[observation["board"] == observation[TEAMMATE].value] = 1

        self._enemies_map = np.zeros(BOARD_SIZE)
        self._enemies_map[observation["board"] == observation[ENEMIES][0].value] = 1
        self._enemies_map[observation["board"] == observation[ENEMIES][1].value] = 1

        # historical positions
        self._agent_history_map = np.where(self._agent_history_map > 0.1, self._agent_history_map - 0.1, 0)
        self._agent_history_map[observation["position"][0], observation["position"][1]] = 1

        self._teammate_history_map = np.where(self._teammate_history_map > 0.1, self._teammate_history_map - 0.1, 0)
        self._teammate_history_map[observation["board"] == observation[TEAMMATE].value] = 1

        self._enemies_history_map = np.where(self._enemies_history_map > 0.1, self._enemies_history_map - 0.1, 0)
        self._enemies_history_map[observation["board"] == observation[ENEMIES][0].value] = 1
        self._enemies_history_map[observation["board"] == observation[ENEMIES][1].value] = 1


    def _update_fog_map(self, observation):
        # just a fucking fog map
        self._fog_map = np.where(observation["board"] == FOG, 1, 0)


    def get_features(self, observation, messages=((0, 0), (0, 0))):
        self._update_materials_map(observation, self._wood_map, WOOD)
        self._update_materials_map(observation, self._stone_map, STONE)
        self._update_materials_map(observation, self._ammo_powerup_map, AMMO_POWERUP)
        self._update_materials_map(observation, self._range_powerup_map, RANGE_POWERUP)
        self._update_materials_map(observation, self._kick_powerup_map, KICK_POWERUP)
        self._update_players_map(observation)
        self._update_fog_map(observation)
        self._update_bomb_map(observation)
        self._update_flame_map(observation)
        self._update_blast_strength_maps()
        self._update_status_maps(observation)
        if self.CN is not None:
            self._update_chat_features_map(messages)

        self._features[:, :, :, 0] = self._wood_map
        self._features[:, :, :, 1] = self._stone_map
        self._features[:, :, :, 2] = self._ammo_powerup_map
        self._features[:, :, :, 3] = self._range_powerup_map
        self._features[:, :, :, 4] = self._kick_powerup_map
        self._features[:, :, :, 5] = self._agent_map
        self._features[:, :, :, 6] = self._teammate_map
        self._features[:, :, :, 7] = self._enemies_map
        self._features[:, :, :, 8] = self._agent_history_map
        self._features[:, :, :, 9] = self._teammate_history_map
        self._features[:, :, :, 10] = self._enemies_history_map
        self._features[:, :, :, 11] = self._fog_map
        self._features[:, :, :, 12] = self._bomb_map
        self._features[:, :, :, 13] = self._bomb_history_map
        self._features[:, :, :, 14] = self._flame_map_1
        self._features[:, :, :, 15] = self._flame_map_2
        self._features[:, :, :, 16] = self._flame_map_3
        self._features[:, :, :, 17] = self._blast_strength_map_1
        self._features[:, :, :, 18] = self._blast_strength_map_2
        self._features[:, :, :, 19] = self._blast_strength_map_3
        self._features[:, :, :, 20] = self._blast_strength_map_4
        self._features[:, :, :, 21] = self._blast_strength_map_5
        self._features[:, :, :, 22] = self._blast_strength_map_6
        self._features[:, :, :, 23] = self._blast_strength_map_7
        self._features[:, :, :, 24] = self._blast_strength_map_8
        self._features[:, :, :, 25] = self._blast_strength_map_9
        self._features[:, :, :, 26] = self._ammo1_map
        self._features[:, :, :, 27] = self._ammo2_map
        self._features[:, :, :, 28] = self._ammo3_map
        self._features[:, :, :, 29] = self._ammo4_map
        self._features[:, :, :, 30] = self._blast1_map
        self._features[:, :, :, 31] = self._blast2_map
        self._features[:, :, :, 32] = self._kick_map
        self._features[:, :, :, 33:37] = self._chat_features_map

        return self._features
