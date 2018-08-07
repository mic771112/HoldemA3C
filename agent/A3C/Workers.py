import numpy as np
import gym
import holdem
import agent
import time
from random import shuffle
from ACNet import ACNet, PLAYER_CACHES_INIT
import json
import card
import datetime

class Worker:
    def __init__(self, mother, name, sess, globalmodel, con_a_opt, dis_a_opt, c_opt, learning=True):

        self.mother = mother
        self.sess = sess
        self.name = name
        self.AC = ACNet(mother, name, sess, globalmodel, con_a_opt, dis_a_opt, c_opt, training=True)
        self.learning = learning
        self.myseat = None
        self.model_list = None
        self.seats = None
        self.env = None
        self.round_start_stack = None
        self.learnable_agent = None


    def webwork(self, opposite_agents, max_global_ep, update_iter, gamma, dump_global_iter, name, uri='ws://poker-training.vtr.trendnet.org:3001/', oppositenum=9):

        self.mother.final_dumped = False
        if 'battle' not in uri:
            init_sleep = np.random.randint(100)
            print('{}\t{}: good day, sir! but let me sleep {}sec more...'.format(self.name, name, init_sleep))
            time.sleep(init_sleep)
        local_game_count = 0
        last_game = datetime.datetime.now()
        while not self.mother.coord.should_stop() and self.mother.global_ep < max_global_ep:  # single move in this loop is a game == a episode
            if 'battle' not in uri:
                game_sleep = np.random.randint(1, 30)
                print('{}\t{}: I am ready, sir! but let me prepare {}sec more...'.format(self.name, name, game_sleep))
                time.sleep(game_sleep)

            # name = 'omg' + str((int(str(self.name)[-1])+self.mother.web_shift) % 16)
            try:
                print('{}\tconnect to {} as {}'.format(self.name, uri, name))
                client_player = holdem.ClientPlayer(uri, name, self.AC, debug=False, playing_live=False)
                client_player.doListen()

                if ('battle' in uri) and ((datetime.datetime.now()-last_game).total_seconds()<10):
                    sec = 100
                    print('{}\t{} is sleeping for {} secs at {}....'.format(self.name, name, sec, uri))
                    time.sleep(sec)
                    last_game = datetime.datetime.now()
                    continue

                local_game_count += 1
            except:
                local_game_count -= 1
                if 'battle' not in uri:
                    sec = 10
                    print('{}\t{} is sleeping for {} secs at {}....'.format(self.name, name, sec, uri))
                    time.sleep(sec)
                else:
                    sec = 3
                    print('{}\t{} is sleeping for {} secs at {}....'.format(self.name, name, sec, uri))
                    time.sleep(sec)

    @staticmethod
    def check_repeat_round_card(state):
        round_card = [i for i in state.community_card if i > 0]
        for p in state.player_states:
            round_card.extend([i for i in p.hand if i > 0])

        if len(round_card) == len(set(round_card)):
            return False
        else:
            print(sorted(round_card))
            for p in state.player_states:
                print(card.deuces2cards([i for i in p.hand if i > 0]))
            print(card.deuces2cards([i for i in state.community_card if i > 0]))

            return True

    def work(self, opposite_agents, max_global_ep, update_iter, gamma, dump_global_iter, uri=None, oppositenum=9):
        self.mother.final_dumped = False
        local_game_count = 0
        local_round_count = 0
        geterror = False
        state = None

        while not self.mother.coord.should_stop() and self.mother.global_ep < max_global_ep:  # single move in this loop is a game == a episode

            local_game_count += 1

            self.env_init(opposite_agents, oppositenum)

            if geterror:
                for seat, agnet in self._get_learnable_agent(self.model_list).items():
                    # agnet.init_cache()
                    agnet.init_learning_buffer()
                    agnet.init_r()

            geterror = False
            game_round = 0
            # print('\ngame start')
            assert not self.env.episode_end
            while not self.env.episode_end:  # single move in this loop is a round == a cycle

                if geterror:
                    print('geterror', geterror, 'check')
                    break
                game_round += 1
                local_round_count += 1
                state = self.env.reset()
                # print('round start')
                cycle_terminal = False

                for s, a in self.learnable_agent.items():
                    a.new_round(state, s)

                assert not cycle_terminal
                while not cycle_terminal:  # single move in this loop is a action

                    geterror = self.check_repeat_round_card(state)
                    if geterror:
                        print(self.name, '====> repeat_round_card!')
                        break


                    # if self.name == 'W_0':
                    #     self.env.render()
                    # self.env._debug = True
                    current_player = state.community_state.current_player
                    actions = holdem.model_list_action(state, n_seats=self.env.n_seats, model_list=self.model_list)

                    try:
                        state, rews, cycle_terminal, info = self.env.step(actions)
                    except (ValueError, gym.error.Error, KeyError) as e:
                        print('=====>', e)
                        geterror = True
                        break

                if geterror:
                    print('====> repeat_round_card!')
                    continue

                for s, a in self.learnable_agent.items():
                    a.round_end(state, s)
                state = self.env.reset()

            if geterror:
                print('====> repeat_round_card!')
                continue

            for s, a in self.learnable_agent.items():
                a.game_over(state, s)

            local_round_count = 0


    @staticmethod
    def _get_stack(state):
        return [i.stack for i in state.player_states]

    # def get_round_reward(self, next_stack):
    #     rewards = [i - j for i, j in zip(next_stack, self.round_start_stack)]
    #     return rewards

    def update_round_start_stack(self, next_stack):
        self.round_start_stack = next_stack

    @staticmethod
    def _get_learnable_agent(model_list):
        return {i: model for i, model in enumerate(model_list) if hasattr(model, 'update_buffer')}

    # def update_player_winning_cache(self, rewards):
    #     learnableagents = self._get_learnable_agent(self.model_list)
    #     for seat, agent in learnableagents.items():
    #         for s, pr in enumerate(rewards):
    #             agent.player_cache[s].update({'winned': 0 if not pr else (pr/abs(pr))})

    # def init_agent_player_cache(self):
    #     learnableagents = self._get_learnable_agent(self.model_list)
    #     for seat, agent in learnableagents.items():
    #         agent.player_cache = PLAYER_CACHES_INIT.copy()

    # def agents_learning(self):
    #     # print('learning')
    #     learnableagents = self._get_learnable_agent(self.model_list)
    #     for seat, agent in learnableagents.items():
    #         if agent.buffer_s:
    #             buffer_s = np.array(agent.buffer_s)
    #             buffer_action = np.array(agent.buffer_action)
    #             buffer_amount = np.array(agent.buffer_amount)
    #             buffer_v = np.array(agent.buffer_v)
    #
    #             feed_dict = {
    #                 agent.s: buffer_s,
    #                 agent.dis_a_his: buffer_action,
    #                 agent.con_a_his: buffer_amount,
    #                 agent.v_target: buffer_v,
    #             }
    #
    #             agent.update_global(feed_dict)
    #             agent.pull_global()
    #             agent.init_learning_buffer()


    # def agents_pull(self):
    #     # print('learning')
    #     learnableagents = self._get_learnable_agent(self.model_list)
    #     for seat, agent in learnableagents.items():
    #         agent.pull_global()

    def env_init(self, opposite_agents, oppositenum):

        shuffle(opposite_agents)
        self.model_list = opposite_agents.copy()[:oppositenum]
        self.seats = oppositenum + 1
        self.myseat = np.random.randint(self.seats)
        self.model_list.insert(self.myseat, self.AC)
        self.env = gym.make('TexasHoldem-v2')  # holdem.TexasHoldemEnv(self.seats)
        self.learnable_agent = self._get_learnable_agent(self.model_list)

        # gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(self.seats)  #  holdem.TexasHoldemEnv(2)
        for i in range(self.seats):
            self.env.add_player(i, stack=3000)
        self.round_start_stack = self._get_stack(self.env.reset())


