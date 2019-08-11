import numpy as np
import time
from constants import *


class FeatureEngineer:
    _wood_map = np.zeros(BOARD_SIZE)
    _stone_map = np.zeros(BOARD_SIZE)
    _ammo_powerup_map = np.zeros(BOARD_SIZE)
    _range_powerup_map = np.zeros(BOARD_SIZE)
    _kick_powerup_map = np.zeros(BOARD_SIZE)
    _fog_map = np.zeros(BOARD_SIZE)

    _agent_map = np.zeros(BOARD_SIZE)
    _teammate_map = np.zeros(BOARD_SIZE)
    _enemies_map = np.zeros(BOARD_SIZE)

    _bomb_map = np.zeros(BOARD_SIZE)
    _bomb_history_map = np.zeros(BOARD_SIZE)
    _hidden_blast_strength_map = np.zeros(BOARD_SIZE)
    _blast_strength_map = np.zeros(BOARD_SIZE)
    _flame_map = np.zeros(BOARD_SIZE)

    _ammo1_map = np.zeros(BOARD_SIZE)
    _ammo2_map = np.zeros(BOARD_SIZE)
    _ammo3_map = np.zeros(BOARD_SIZE)
    _ammo4_map = np.zeros(BOARD_SIZE)
    _blast1_map = np.zeros(BOARD_SIZE)
    _blast2_map = np.zeros(BOARD_SIZE)
    _blast3_map = np.zeros(BOARD_SIZE)
    _kick_map = np.zeros(BOARD_SIZE)

    _features = np.zeros((1, BOARD_SIZE[0], BOARD_SIZE[1], N_FEATURES), dtype="float32")

    # _agent_number = 0
    #
    # def _get_agent_number(self, observation):
    #     if not self._agent_number:
    #         self._agent_number = 10 + 11 + 12 + 13 - observation[TEAMMATE].value - observation[ENEMIES][0].value - observation[ENEMIES][1].value
    #     return self._agent_number

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
        # ammo1, ammo2, ammo3, ammo4, blast_strength2, blast_strength3, blast_strength4, can_kick, 0, 0, 0
        self._ammo1_map = np.ones(BOARD_SIZE) if observation["ammo"] > 0 else np.zeros(BOARD_SIZE)
        self._ammo2_map = np.ones(BOARD_SIZE) if observation["ammo"] > 1 else np.zeros(BOARD_SIZE)
        self._ammo3_map = np.ones(BOARD_SIZE) if observation["ammo"] > 2 else np.zeros(BOARD_SIZE)
        self._ammo4_map = np.ones(BOARD_SIZE) if observation["ammo"] > 3 else np.zeros(BOARD_SIZE)
        self._blast1_map = np.ones(BOARD_SIZE) if observation["blast_strength"] > 1 else np.zeros(BOARD_SIZE)
        self._blast2_map = np.ones(BOARD_SIZE) if observation["blast_strength"] > 2 else np.zeros(BOARD_SIZE)
        self._blast3_map = np.ones(BOARD_SIZE) if observation["blast_strength"] > 3 else np.zeros(BOARD_SIZE)
        self._kick_map = np.ones(BOARD_SIZE) if int(observation["can_kick"]) else np.zeros(BOARD_SIZE)

    # should be used for WOOD, STONE, POWERUPS, has be updated before flames
    def _update_materials_map(self, observation, map, material):

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["board"][top:bottom, left:right]

        # maps the squares in fov into 1/0 depending on the chosen material
        filtered_fov = np.where(fov == material, 1, 0)

        # rewrites visible part of the map with updated mappings
        map[top:bottom, left:right] = filtered_fov

        # remove wood at the end of flames (using flames map from t - 1)
        if material == WOOD:
            map[self._flame_map == 0.1] = 0

    # has be updated before blast strength map
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
        # it takes blast strenght map from previous timestep and maps exploded squares to 0.9
        # it then takes values of 0.9, 0.3 or 0.1 based on flame time left
        self._flame_map = np.where(self._flame_map > 0.1, self._flame_map / 3, 0)
        self._flame_map[self._blast_strength_map == 0.9] = 0.9

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["flame_life"][top:bottom, left:right]

        # maps the squares in fov into 0.9, 0.3, 0.1 and 0
        filtered_fov = np.where(fov > 0, 0.9 / 3 ** (3 - fov), 0)

        # rewrites visible part of the map with updated mappings
        self._flame_map[top:bottom, left:right] = filtered_fov

    def _update_blast_strength_map(self):
        # blast strength map always starts with a clean slate and just maps the current bomb map
        # to the expected explosion radius based on a hidden map called _hidden_blast_strength_map
        # (it calculates chained explosions etc, and the squares' values are a count up from 0.1 to 0.9)
        # (1 represents the explosion and is not shown on this map

        # clear explosions that happened unexpectedly
        self._blast_strength_map = np.zeros(BOARD_SIZE)

        row = np.where(self._bomb_map > 0)[0]
        col = np.where(self._bomb_map > 0)[1]

        # [row, col, blast strength, life]
        bombs = []
        for i in range(len(col)):
            bombs.append([row[i], col[i], int(self._hidden_blast_strength_map[row[i], col[i]] - 1),
                          self._bomb_map[row[i], col[i]]])
        bombs.sort(key=lambda tup: tup[3], reverse=True)

        # print(bombs)
        # creates expected blast radius & time for all bombs (including chaining explosions etc)
        while bombs:
            bomb_range = bombs[0][2]
            bomb_life = bombs[0][3]
            self._blast_strength_map[bombs[0][0], bombs[0][1]] = bomb_life

            # writes bomb life to _blast_strength_map
            # and if there is a bomb on this square, it changes its' bomb life in the bombs array
            def _check4bomb(row, col, bomb_life):
                self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    # print(self._bomb_map)
                    # print(np.where(self._bomb_map > 0))
                    # print(bombs, flush=True)
                    # print(row, col)
                    i = [i for i in range(len(bombs)) if bombs[i][0] == row and bombs[i][1] == col]
                    if i:
                        bombs[i[0]][3] = bomb_life

            row, col = bombs[0][0], bombs[0][1] + 1
            a = 0
            # while in the map and in range and not on a stone square and while the previous square wasn't wood
            # right
            while col < BOARD_SIZE[1] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[
                row, col - 1] != 1:
                _check4bomb(row, col, bomb_life)
                col += 1
                a += 1

            row, col = bombs[0][0], bombs[0][1] - 1
            a = 0
            # left
            while col >= 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row, col + 1] != 1:
                _check4bomb(row, col, bomb_life)
                col -= 1
                a += 1

            row, col = bombs[0][0] + 1, bombs[0][1]
            a = 0
            # down
            while row < BOARD_SIZE[0] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[
                row - 1, col] != 1:
                _check4bomb(row, col, bomb_life)
                row += 1
                a += 1

            row, col = bombs[0][0] - 1, bombs[0][1]
            a = 0
            # up
            while row >= 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row + 1, col] != 1:
                _check4bomb(row, col, bomb_life)
                row -= 1
                a += 1

            bombs.pop(0)

    def _update_players_map(self, observation, player_map, player):
        # player maps are maps where the known position of a player is represented by "1" and all other board
        # values are discounted by PLAYER_DECAY every round no matter what
        # This is a controversial decision and these "historical values" might be turned off entirely in the future
        player_map *= PLAYER_DECAY
        if player == AGENT:
            # player_map = np.zeros(BOARD_SIZE)
            player_map[observation["position"][0], observation["position"][1]] = 1
        if player == TEAMMATE:
            player_map[observation["board"] == observation[TEAMMATE].value] = 1
        if player == ENEMIES:
            player_map[observation["board"] == observation[ENEMIES][0].value] = 1
            player_map[observation["board"] == observation[ENEMIES][1].value] = 1

    def _update_fog_map(self, observation):
        # just a fucking fog map
        self._fog_map = np.where(observation["board"] == FOG, 1, 0)

    def get_features(self, observation):
        self._update_materials_map(observation, self._wood_map, WOOD)
        self._update_materials_map(observation, self._stone_map, STONE)
        self._update_materials_map(observation, self._ammo_powerup_map, AMMO_POWERUP)
        self._update_materials_map(observation, self._range_powerup_map, RANGE_POWERUP)
        self._update_materials_map(observation, self._kick_powerup_map, KICK_POWERUP)
        self._update_players_map(observation, self._enemies_map, ENEMIES)
        self._update_players_map(observation, self._teammate_map, TEAMMATE)
        self._update_players_map(observation, self._agent_map, AGENT)
        self._update_fog_map(observation)
        self._update_bomb_map(observation)
        self._update_flame_map(observation)
        self._update_blast_strength_map()
        self._update_status_maps(observation)

        self._features[:, :, :, 0] = self._wood_map
        self._features[:, :, :, 1] = self._stone_map
        self._features[:, :, :, 2] = self._ammo_powerup_map
        self._features[:, :, :, 3] = self._range_powerup_map
        self._features[:, :, :, 4] = self._kick_powerup_map
        self._features[:, :, :, 5] = self._enemies_map
        self._features[:, :, :, 6] = self._teammate_map
        self._features[:, :, :, 7] = self._agent_map
        self._features[:, :, :, 8] = self._fog_map
        self._features[:, :, :, 9] = self._bomb_map
        self._features[:, :, :, 10] = self._bomb_history_map
        self._features[:, :, :, 11] = self._flame_map
        self._features[:, :, :, 12] = self._blast_strength_map
        self._features[:, :, :, 13] = self._ammo1_map
        self._features[:, :, :, 14] = self._ammo2_map
        self._features[:, :, :, 15] = self._ammo3_map
        self._features[:, :, :, 16] = self._ammo4_map
        self._features[:, :, :, 17] = self._blast1_map
        self._features[:, :, :, 18] = self._blast2_map
        self._features[:, :, :, 19] = self._blast3_map
        self._features[:, :, :, 20] = self._kick_map
        self._features[:, :, :, 21] = np.zeros(BOARD_SIZE)
        self._features[:, :, :, 22] = np.zeros(BOARD_SIZE)
        self._features[:, :, :, 23] = np.zeros(BOARD_SIZE)
        self._features[:, :, :, 24] = np.zeros(BOARD_SIZE)

        return self._features
