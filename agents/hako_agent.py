#################################
# (C) Copyright IBM Corp. 2018
#################################
import os
import sys
import random
import array
from subprocess import check_output, call
import time
from threading import Thread
from py4j.java_gateway import JavaGateway, GatewayParameters
from pommerman import characters
from pommerman.agents import BaseAgent
verbose = False


class HakoAgent(BaseAgent):

    def __init__(self, port):
        super(HakoAgent, self).__init__(characters.Bomber)
        self.path = os.path.abspath("agents/hako/BBMServer" + str(port) + ".jar")
        self.server_thread = Thread(target=self.start_server, daemon=True)
        self.server_thread.start()
        time.sleep(5)
        self._pid = os.getpid()
        self._me = -1
        self._caller_id = random.randint(1000000, 10000000)
        # print("MyAgent.__init__, pid={}, caller_id={}, me={}".format(self._pid, self._caller_id, self._me), flush=True)
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
        self._addition_app = self.gateway.entry_point
        self._addition_app.init_agent(self._pid, self._caller_id, self._me)
        print("agent initialized", flush=True)

    def start_server(self):
        check_output("java -jar " + self.path, shell=True)

    def episode_end(self, reward):
        # print("MyAgent.episode_end, pid={}, caller_id={}, me={}".format(self._pid, self._caller_id, self._me), flush=True)
        self._addition_app.episode_end(self._pid, self._caller_id, self._me, reward)





    def act(self, obs, action_space):

        ##########################################################################################################################################
        ##########################################################################################################################################
        ##########################################################################################################################################
        ##########################################################################################################################################

        # print("MyAgent.act, pid={}, caller_id={}, me={}".format(self._pid, self._caller_id, self._me), flush=True)


        def pack_into_buffer(data):
            data.flatten()
            data.flatten().tolist()
            header = array.array('i', list(data.shape))
            body = array.array('d', data.flatten().tolist())
            if sys.byteorder != 'big':
                header.byteswap()
                body.byteswap()
            buffer = bytearray(header.tostring() + body.tostring())
            return buffer

        def pack_into_buffer2(data):
            header = array.array('i', [len(data), 1])
            body = array.array('d', data)
            if sys.byteorder != 'big':
                header.byteswap()
                body.byteswap()
            buffer = bytearray(header.tostring() + body.tostring())
            return buffer


        isCollapse = obs['game_env']=='pommerman.envs.v1:Pomme'

        # pick up all data from obs.
        position = obs['position']
        ammo = (int)(obs['ammo'])
        blast_strength = (int)(obs['blast_strength'])
        can_kick = (bool)(obs['can_kick'])

        board = obs['board']
        bomb_blast_strength = obs['bomb_blast_strength']
        bomb_life = obs['bomb_life']

        alive = obs['alive']
        enemies = obs['enemies']
        teammate = obs['teammate']



        # reshape array, list, or other non-primitive objects into byte array objects to send it to Java.
        x = (int)(position[0])
        y = (int)(position[1])

        self._me = (int)(board[x][y])

        board_buffer = pack_into_buffer(board)
        bomb_blast_strength_buffer = pack_into_buffer(bomb_blast_strength)
        bomb_life_buffer = pack_into_buffer(bomb_life)

        alive_buffer = pack_into_buffer2(alive)

        enemies_list = []
        for enemy in enemies:
            enemies_list.append(enemy.value)
        enemies_list_buffer = pack_into_buffer2(enemies_list)


        # call Java function.
        action = self._addition_app.act(self._pid, self._caller_id, self._me, x, y, ammo, blast_strength, can_kick, board_buffer, bomb_blast_strength_buffer, bomb_life_buffer, alive_buffer, enemies_list_buffer, teammate.value, isCollapse)
        return action

        ##########################################################################################################################################
        ##########################################################################################################################################
        ##########################################################################################################################################
        ##########################################################################################################################################

    def shutdown(self):
        self.gateway.shutdown()
        call("kill -9 $(lsof -t " + self.path + ") &> /dev/null", shell=True)
        time.sleep(5)
