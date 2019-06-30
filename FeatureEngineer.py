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
    _hidden_blast_strength_map = np.zeros(BOARD_SIZE)
    _blast_strength_map = np.zeros(BOARD_SIZE)

    # _agent_number = 0
    #
    # def _get_agent_number(self, observation):
    #     if not self._agent_number:
    #         self._agent_number = 10 + 11 + 12 + 13 - observation[TEAMMATE].value - observation[ENEMIES][0].value - observation[ENEMIES][1].value
    #     return self._agent_number

    def _get_fov_boundries(self, observation):
        # player position
        y = observation["position"][0]
        x = observation["position"][1]
        # players field of view (clipped)
        top = max(0, y - 4)
        bottom = min(y + 5, BOARD_SIZE[0])
        left = max(0, x - 4)
        right = min(x + 5, BOARD_SIZE[1])
        return top, bottom, left, right

    # should be used for WOOD, STONE, POWERUPS
    def _update_materials_map(self, observation, map, material):
        # gets boundries of agents' field of view
        top, bottom, left, right = self._get_fov_boundries(observation)
        # gets the visible part of the board
        fov = observation["board"][top:bottom, left:right]
        # maps the squares in fov into 1/0 depending on the chosen material
        filtered_fov = np.where(fov == material, 1, 0)
        # rewrites visible part of the map with updated mappings
        map[top:bottom, left:right] = filtered_fov

    def _update_bomb_map(self, observation):
        self._bomb_map[self._bomb_map > 0] += 0.1
        self._bomb_map[self._bomb_map > 1] = 0
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

    def _update_blast_strength_map(self, observation):
        print(np.where(self._bomb_map > 0)[0])
        print(np.where(self._bomb_map > 0)[1])
        # print("------------------")
        # print(self._bomb_map)
        pass

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
        self._update_blast_strength_map(observation)
        # print(time.time() - tim)
        # print("---------------------------------------------------------------"))