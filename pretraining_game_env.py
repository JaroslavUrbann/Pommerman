import pommerman
from pommerman import agents
from docker_agent import DockerAgent
import pretraining_database
from constants import *
import time
import subprocess


def create_database(database_size):
    # print(pommerman.REGISTRY)

    pretraining_database.create_database(database_size)

    for i in range(4):
        # just to prevent re-initializing dockers 4x when using small db size
        if (i == 0 and pretraining_database.step_index < database_size / 4) or (
                i == 1 and pretraining_database.step_index < database_size / 2) or (
                i == 2 and pretraining_database.step_index < 3 * database_size / 4) or (
                i == 3 and pretraining_database.step_index < database_size):
            hako1 = DockerAgent("multiagentlearning/hakozakijunctions", 80, 1)
            hako2 = DockerAgent("multiagentlearning/hakozakijunctions", 81, 2)
            skynet1 = DockerAgent("multiagentlearning/skynet955", 82, 3)
            skynet2 = DockerAgent("multiagentlearning/skynet955", 83, 4)
            if i == 0:
                agent_list = [hako1, skynet1, hako2, skynet2]
            if i == 1:
                agent_list = [hako2, skynet1, hako1, skynet2]
            if i == 2:
                agent_list = [skynet1, hako1, skynet2, hako2]
            if i == 3:
                agent_list = [skynet1, hako2, skynet2, hako1]
            env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        # Run the episodes just like OpenAI Gym
        while (i == 0 and pretraining_database.step_index < database_size / 4) or (
                i == 1 and pretraining_database.step_index < database_size / 2) or (
                i == 2 and pretraining_database.step_index < 3 * database_size / 4) or (
                i == 3 and pretraining_database.step_index < database_size):
            state = env.reset()
            done = False
            while not done:
                env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                pretraining_database.step_index += 1
            print("database items done: " + str(pretraining_database.step_index))
        env.close()
        subprocess.call('docker kill $(docker ps -q)', shell=True)
