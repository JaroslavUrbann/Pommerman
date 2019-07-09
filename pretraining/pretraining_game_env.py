import pommerman
from pretraining import pretraining_database
import subprocess
from pretraining.hako_agent import HakoAgent
from pommerman.agents import SimpleAgent
import time

def fill_database():
    # print(pommerman.REGISTRY)
    tim = time.time()
    db_size = pretraining_database.database_size
    ng = False
    for i in range(2):
        # just to prevent re-initializing dockers 4x when using small db size
        if (i == 0 and pretraining_database.step_index < db_size / 2) or (
                i == 1 and pretraining_database.step_index < db_size):
            ng = True
            hako1 = HakoAgent(a_id=1)
            hako2 = HakoAgent(a_id=2)
            hako3 = HakoAgent(a_id=3)
            # hako1 = SimpleAgent()
            # hako2 = SimpleAgent()
            # hako3 = SimpleAgent()
            # hako4 = SimpleAgent()
            hako4 = HakoAgent(a_id=4)
            if i == 0:
                agent_list = [hako1, hako3, hako2, hako4]
            if i == 1:
                agent_list = [hako2, hako4, hako1, hako3]
            env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        while (i == 0 and pretraining_database.step_index < db_size / 2) or (
                i == 1 and pretraining_database.step_index < db_size):
            state = env.reset()
            done = False
            while not done:
                # env.render()
                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                # getting 2 new values for x1 and x2 every step because I'm getting data from all 4 players
                pretraining_database.step_index += 2
            print("database items done: " + str(pretraining_database.step_index) + "/" + str(db_size) + " i = " + str(i))
        if ng:
            env.close()
        ng = False
        # subprocess.call('docker kill $(docker ps -q)', shell=True)
    print("filling db done in: " + str(time.time() - tim))
