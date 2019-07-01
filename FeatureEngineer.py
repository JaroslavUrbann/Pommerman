import numpy as np
import time

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
PLAYER_DECAY = 0.8
BOMB = 3


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

    _status_map = np.zeros(BOARD_SIZE)

    id = 0
    xd = 0

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

    def _update_status_map(self, observation):
        # ammo1, ammo2, ammo3, ammo4, blast_strength2, blast_strength3, blast_strength4, can_kick, 0, 0, 0
        top_line = np.zeros(BOARD_SIZE[1])
        for i in range(4):
            if observation["ammo"] > i:
                top_line[i] = 1
        for i in range(1, 4):
            if observation["blast_strength"] > i:
                top_line[i + 3] = 1
        top_line[7] = int(observation["can_kick"])
        self._status_map[0] = top_line


    # should be used for WOOD, STONE, POWERUPS, should be updated before flames
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

    def _update_bomb_map(self, observation):
        self._bomb_history_map = self._bomb_map.copy()

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
        self._hidden_blast_strength_map[top:bottom, left:right] = observation["bomb_blast_strength"][top:bottom, left:right]

    # has to be updated before blast strength map because it needs to work with blast strength map at T-1
    def _update_flame_map(self, observation):
        self._flame_map = np.where(self._flame_map > 0.1, self._flame_map / 3, 0)
        self._flame_map[self._blast_strength_map == 0.9] = 0.9

        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)

        # gets the visible part of the board
        fov = observation["flame_life"][top:bottom, left:right]

        # maps the squares in fov into 0.9, 0.3, 0.1 and 0
        filtered_fov = np.where(fov > 0, 0.9 / 3**(3 - fov), 0)

        # rewrites visible part of the map with updated mappings
        self._flame_map[top:bottom, left:right] = filtered_fov

    # has to be updated only after bomb map
    def _update_blast_strength_map(self):
        # clear explosions that happened unexpectedly
        self._blast_strength_map = np.zeros(BOARD_SIZE)

        row = np.where(self._bomb_map > 0)[0]
        col = np.where(self._bomb_map > 0)[1]

        # (row, col, blast strength, life)
        bombs = []
        for i in range(len(col)):
            bombs.append((row[i], col[i], int(self._hidden_blast_strength_map[row[i], col[i]] - 1), self._bomb_map[row[i], col[i]]))
        bombs.sort(key=lambda tup: tup[3], reverse=True)

        # creates expected blast radius & time for all bombs (including chaining explosions etc)
        while bombs:
            bomb_range = bombs[0][2]
            bomb_life = bombs[0][3]
            self._blast_strength_map[bombs[0][0], bombs[0][1]] = bomb_life

            row, col = bombs[0][0], bombs[0][1] + 1
            a = 0
            while col < BOARD_SIZE[1] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row, col - 1] != 1:
                self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    bombs[np.where(np.all(bombs[0] == row, bombs[1] == col))[0]][3] = bomb_life
                col += 1
                a += 1

            row, col = bombs[0][0], bombs[0][1] - 1
            a = 0
            while col > 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row, col + 1] != 1:
                self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    bombs[np.where(np.all(bombs[0] == row, bombs[1] == col))[0]][3] = bomb_life
                col -= 1
                a += 1

            row, col = bombs[0][0] + 1, bombs[0][1]
            a = 0
            while col < BOARD_SIZE[0] and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row - 1, col] != 1:
                self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    bombs[np.where(np.all(bombs[0] == row, bombs[1] == col))[0]][3] = bomb_life
                row += 1
                a += 1

            row, col = bombs[0][0] - 1, bombs[0][1]
            a = 0
            while col > 0 and self._stone_map[row, col] != 1 and bomb_range > a and self._wood_map[row + 1, col] != 1:
                self._blast_strength_map[row, col] = bomb_life
                if self._bomb_map[row, col] > 0:
                    bombs[np.where(np.all(bombs[0] == row, bombs[1] == col))[0]][3] = bomb_life
                row -= 1
                a += 1

            bombs.pop(0)

    def _update_players_map(self, observation, map, player):
        map *= PLAYER_DECAY
        if player == AGENT:
            map[observation["position"][0], observation["position"][1]] = 1
        if player == TEAMMATE:
            map[observation["board"] == observation[TEAMMATE].value] = 1
        if player == ENEMIES:
            map[observation["board"] == observation[ENEMIES][0].value] = 1
            map[observation["board"] == observation[ENEMIES][1].value] = 1

    def _update_fog_map(self, observation):
        self._fog_map = np.where(observation["board"] == FOG, 1, 0)

    def make_features(self, observation):
        tim = time.time()
        self._update_players_map(observation, self._enemies_map, ENEMIES)
        self._update_players_map(observation, self._teammate_map, TEAMMATE)
        self._update_players_map(observation, self._agent_map, AGENT)
        self._update_fog_map(observation)
        self._update_materials_map(observation, self._wood_map, WOOD)
        self._update_materials_map(observation, self._stone_map, STONE)
        self._update_materials_map(observation, self._ammo_powerup_map, AMMO_POWERUP)
        self._update_materials_map(observation, self._range_powerup_map, RANGE_POWERUP)
        self._update_materials_map(observation, self._kick_powerup_map, KICK_POWERUP)
        self._update_bomb_map(observation)
        self._update_flame_map(observation)
        self._update_blast_strength_map()
        self._update_status_map(observation)
        # print(self._status_map)
        # print(observation["flame_life"])
        self.xd += time.time() - tim
        self.id += 1
        print(self.xd/self.id)
        # print("---------------------------------------------------------------")
        # print(observation["bomb_blast_strength"])