from pommerman.agents import BaseAgent
from pommerman import characters
from contextlib import contextmanager
import ctypes
import os
import tempfile
import sys


class EisenachAgent(BaseAgent):

    def __init__(self, character=characters.Bomber):
        super().__init__()
        self._character = character
        self.avg_simsteps_per_turns = []

    def __getattr__(self, attr):
        return getattr(self._character, attr)

    def act(self, obs, action_space):
        with captured_stdout() as E:
            decision = self.c.c_getStep_eisenach(
                self.id,
                10 in obs['alive'], 11 in obs['alive'], 12 in obs['alive'], 13 in obs['alive'],
                obs['board'].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                obs['bomb_life'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                obs['bomb_blast_strength'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(obs['position'][0]), int(obs['position'][1]),
                obs['blast_strength'],
                obs['can_kick'],
                obs['ammo'],
                obs['teammate'].value
                )

        return decision

    def episode_end(self, reward):
        # with captured_stdout() as E:
        with suppress_stdout():
            self.c.c_episode_end_eisenach.restype = ctypes.c_float
            avg_simsteps_per_turn = self.c.c_episode_end_eisenach(self.id)
            self.avg_simsteps_per_turns.append(avg_simsteps_per_turn)

    def init_agent(self, id, game_type):
        self.id = id
        self._character = self._character(id, game_type)

        self.c = ctypes.cdll.LoadLibrary(os.path.abspath("agents/eisenach/pommerman_cpp/cmake-build-debug/libmunchen.so"))

        self.c.c_init_agent_eisenach(id)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class captured_stdout:
    def __init__(self):
        self.prevfd = None
        self.prev = None

    def __enter__(self):
        F = tempfile.NamedTemporaryFile()
        self.prevfd = os.dup(sys.stdout.fileno())
        os.dup2(F.fileno(), sys.stdout.fileno())
        self.prev = sys.stdout
        sys.stdout = os.fdopen(self.prevfd, "w")
        return F

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.prevfd, self.prev.fileno())
        sys.stdout = self.prev