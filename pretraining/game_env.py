import pommerman
from pretraining import DB
import subprocess
from agents.hako_agent import HakoAgent
from pommerman.agents import SimpleAgent
import time


def fill_database():
    # print(pommerman.REGISTRY)
    tim = time.time()
    db_size = DB.database_size

    hako1 = HakoAgent(port=25336)
    hako2 = HakoAgent(port=25337)
    hako3 = HakoAgent(port=25338)
    hako4 = HakoAgent(port=25339)

    agent_list = [hako1, hako2, hako3, hako4]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    while DB.step_index < db_size:
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            DB.add_data(state, actions)
            if state[0]['step_count'] > 200:
                break
        DB.next_episode()
        print("database items done: " + str(DB.step_index) + "/" + str(db_size))
    env.close()
        # subprocess.call('docker kill $(docker ps -q)', shell=True)
    print("-----------------------------------------------------------------------------")
    print("filling db done in: " + str(time.time() - tim))
    print("-----------------------------------------------------------------------------")