from PIL import Image
from IPython import display
import pommerman
from pommerman import agents
from docker_agent import DockerAgent
# import matplotlib.pyplot as plt
from pommerman.agents import BaseAgent
from feature_engineer import FeatureEngineer


def main():
    """Simple function to bootstrap a game"""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    #     video_maker = VideoMaker()

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # DockerAgent("multiagentlearning/hakozakijunctions", 80)
        TestAgent(),
        # TestAgent(),
        # TestAgent(),
        # agents.RandomAgent(),
        # agents.RandomAgent(),
        # agents.PlayerAgent(agent_control="arrows")
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    # Run the episodes just like OpenAI Gym
    i = 0
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            i += 1
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            print(info["result"].value)
            # if i == 300:
            #     break
        print('Episode {} finished'.format(i_episode))
    env.close()


class TestAgent(BaseAgent):
    FeatureEngineer = FeatureEngineer()

    def act(self, observation, action_space):

        # print(sys.getsizeof(observation))
        # self.FeatureEngineer.make_features(observation)
        # time.sleep(0.5)
        # print(observation["teammate"].value)
        # print(observation)
        return 0 #random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)


main()
