import pommerman
from agents import LN_agent
from agents.LN_agent import TestingAgent
from agents.docker_agent import DockerAgent
import subprocess


def test_network(n_games, LN):

    LN_agent.LN = LN

    # win / loss / tie
    results = [0, 0, 0]

    for i in range(4):
        agent1 = TestingAgent(1)
        agent2 = TestingAgent(2)
        eisenach1 = DockerAgent("multiagentlearning/eisenach", 81, 0)
        eisenach2 = DockerAgent("multiagentlearning/eisenach", 82, 0)
        if i == 0:
            agent_list = [agent1, eisenach1, agent2, eisenach2]
        if i == 1:
            agent_list = [agent2, eisenach1, agent1, eisenach2]
        if i == 2:
            agent_list = [eisenach1, agent1, eisenach2, agent2]
        if i == 3:
            agent_list = [eisenach1, agent2, eisenach2, agent1]

        env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        for a in range(int(n_games / 4)):
            state = env.reset()
            done = False
            while not done:
                # env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)

            if info["result"].value == 0:
                if info["winners"][0] == 0 and i < 3:
                    results[0] += 1
                else:
                    results[1] += 1
            else:
                results[2] += 1
            print("games done: " + str(int(i * n_games / 4 + a + 1)))
        env.close()
        subprocess.call('docker kill $(docker ps -q)', shell=True)
    return results
