import pommerman
from pretraining import DB
from agents.hako_agent import HakoAgent
from pommerman.agents import SimpleAgent
import time


def fill_database(DB1, DB2, DB3):
    # print(pommerman.REGISTRY)
    tim = time.time()

    hako1 = HakoAgent(port=25336)
    hako2 = HakoAgent(port=25337)
    hako3 = HakoAgent(port=25338)
    hako4 = HakoAgent(port=25339)

    agent_list = [hako1, hako2, hako3, hako4]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    while DB1.step_index < DB1.size or DB2.step_index < DB2.size or DB3.step_index < DB3.size:
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            if state[0]['step_count'] < 60:
                DB1.add_data(state, actions)
            if 60 <= state[0]['step_count'] < 180:
                DB2.add_data(state, actions)
                if DB2.step_index == DB2.size and DB3.step_index == DB3.size:
                    break
            if 180 <= state[0]['step_count']:
                DB3.add_data(state, actions)
                if DB3.step_index == DB3.size:
                    break
            state, reward, done, info = env.step(actions)
        DB1.next_episode()
        DB2.next_episode()
        DB3.next_episode()
        print("database items done: " + str(DB1.step_index) + " | " + str(DB2.step_index) + " | " + str(DB3.step_index) + "/" + str(DB1.size) + " | " + str(DB2.size) + " | " + str(DB3.size) + " in time: " + str(int(time.time() - tim)))
    env.close()
    print("-----------------------------------------------------------------------------")
    print("filling db done in: " + str(time.time() - tim))
    print("-----------------------------------------------------------------------------")

