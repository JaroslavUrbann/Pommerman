import pommerman
from pommerman import agents
# import matplotlib.pyplot as plt
from pommerman.agents import BaseAgent
from feature_engineer import FeatureEngineer
from agents.docker_agent import DockerAgent
import time


def main():
    """Simple function to bootstrap a game"""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    #     video_maker = VideoMaker()

    # Create a set of agents (exactly four)
    agent_list = [
        # agents.SimpleAgent(),
        # DockerAgent("multiagentlearning/navocado", port=80),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # DockerAgent("multiagentlearning/hakozakijunctions", 80)
        # TestAgent(),
        TestAgent(),
        TestAgent(),
        TestAgent(),
        # agents.RandomAgent(),
        # agents.RandomAgent(),
        agents.PlayerAgent(agent_control="arrows")
        # DockerAgent("multiagentlearning/navocado", port=81),
    ]

    # env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    # Run the episodes just like OpenAI Gym
    i = 0
    tim = time.time()
    for i_episode in range(1):
        state = env.reset()
        done = False
        feature_engineer = FeatureEngineer()
        while not done:
            i += 1
            env.render()
            actions = env.act(state)
            # print(state[3]["bomb_life"])
            state, reward, done, info = env.step(actions)
            # print(done)
            xd = feature_engineer.get_features(state[3])
            # print(xd[0, :, :, 9])
            # if i == 300:
            # break
            # print(info)
            # break
        print('Episode {} finished'.format(i_episode))
    env.close()
    print(time.time() - tim)


class TestAgent(BaseAgent):
    feature_engineer = FeatureEngineer()

    def act(self, observation, action_space):
        # xd = self.feature_engineer.get_features(observation)
        # print(xd[0, :, :, 9])
        # print(observation["bomb_life"])
        # print(xd[0, :, :, 0])
        # print(".........")
        # print(xd[0, :, :, 11])
        # print(observation["board"])
        # print("...........................................................")
        # print(observation["flame_life"])
        # print("...........................................................")
        # self.feature_engineer.update_features(observation)
        # print(sys.getsizeof(observation))
        # self.FeatureEngineer.make_features(observation)
        # time.sleep(0.5)
        # print(observation["teammate"].value)
        # print(observation)
        return 0 #random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)

    def episode_end(self, reward):
        print(reward)
main()
