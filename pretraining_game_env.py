import pommerman
from pommerman import agents
from docker_agent import DockerAgent
import pretraining_database
from constants import *


def run_games():
    # print(pommerman.REGISTRY)
    for i in range(4):
        hako1 = DockerAgent("multiagentlearning/hakozakijunctions", 80, 1)
        hako2 = DockerAgent("multiagentlearning/hakozakijunctions", 81, 2)
        skynet1 = DockerAgent("multiagentlearning/skynet955", 82, 0)
        skynet2 = DockerAgent("multiagentlearning/skynet955", 83, 0)
        if i == 0:
            agent_list = [hako1, skynet1, hako2, skynet2]
        if i == 1:
            agent_list = [hako2, skynet1, hako1, skynet2]
        if i == 2:
            agent_list = [skynet1, hako1, skynet2, hako2]
        if i == 3:
            agent_list = [skynet1, hako2, skynet2, hako1]
        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        # Run the episodes just like OpenAI Gym
        while (i == 0 and pretraining_database.step_index < DATABASE_SIZE / 4) or (
                i == 1 and pretraining_database.step_index < DATABASE_SIZE / 2) or (
                i == 2 and pretraining_database.step_index < 3 * DATABASE_SIZE / 4) or (
                i == 3 and pretraining_database.step_index < DATABASE_SIZE):
            state = env.reset()
            done = False
            while not done:
                env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                pretraining_database.step_index += 1
            print(i, pretraining_database.step_index)
        env.close()
