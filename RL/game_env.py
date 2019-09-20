import pommerman
from feature_engineer import FeatureEngineer
from pommerman.agents import BaseAgent
from pommerman import characters
from RL.training import Training
import time


def train_network(models, chat_model, n_steps, max_time):
    tr = time.time()

    T = Training(models, chat_model)

    agent0 = NetworkAgent(0, T)
    agent1 = NetworkAgent(1, T)
    agent2 = NetworkAgent(2, T)
    agent3 = NetworkAgent(3, T)

    agent_list = [agent0, agent1, agent2, agent3]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    n_episodes = 0
    step = 0
    while step < n_steps and max_time > time.time() - tr:
        state = env.reset()
        died_first = []
        alive = state[0]["alive"]
        done = False
        e_step = 0
        while not done:
            tim = time.time()
            actions = env.act(state)
            step += 1
            e_step += 1

            died = list(set(alive) - set(state[0]["alive"]))
            # max 1 person from each team
            died_first += [d-10 for d in died if ((d - 8) % 4) not in died_first]

            alive = state[0]["alive"]
            state, reward, done, info = env.step(actions)
            T.end_step()
#             print(e_step, "time:", time.time() - tim, flush=True)
        n_episodes += 1
        T.end_episode(n_episodes, died_first, info["winners"] if not info["result"] else [])
        print(n_episodes, "steps:", step, "time:", int(time.time() - tr), info, flush=True)
    env.close()
    print("----------------------------------------------------------------------------------------------")
    print("RL training done in: " + str(time.time() - tr) + " n_episodes: " + str(n_episodes) + " n_steps: " + str(step))
    print("----------------------------------------------------------------------------------------------")


class NetworkAgent(BaseAgent):

    def __init__(self, a_id, T):
        super(NetworkAgent, self).__init__(characters.Bomber)
        self.feature_engineer = FeatureEngineer()
        self.a_id = a_id
        self.T = T

    def act(self, observation, action_space):
        features = self.feature_engineer.get_features(observation)
        action = self.T.training_step(features, self.a_id, observation["position"])
        return action, 0, 0

    def episode_end(self, reward):
        self.feature_engineer = FeatureEngineer()