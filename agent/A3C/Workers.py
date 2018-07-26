import numpy as np
import gym
import holdem
import agent
import time
from random import shuffle
from ACNet import ACNet, PLAYER_CACHES_INIT
import json
import card

class Worker:
    def __init__(self, mother, name, sess, globalmodel, a_opt, c_opt, learning=True):
        self.mother = mother
        self.sess = sess
        self.name = name
        self.AC = ACNet(mother, name, sess, globalmodel, a_opt, c_opt, training=True)
        self.learning = learning
        self.myseat = None
        self.model_list = None
        self.seats = None
        self.env = None
        self.round_start_stack = None


    def webwork(self, opposite_agents, max_global_ep, update_global_iter, gamma, dump_global_iter, uri='ws://poker-training.vtr.trendnet.org:3001/', oppositenum=9):
        self.mother.final_dumped = False
        time.sleep(np.random.randint(100))
        local_game_count = 0
        local_round_count = 0
        # self.env_init(opposite_agents)

        # self.learnable_agent = self._get_learnable_agent(self.model_list)

        while not self.mother.coord.should_stop() and self.mother.global_ep < max_global_ep:  # single move in this loop is a game == a episode
            time.sleep(np.random.randint(5, 20))
            local_game_count += 1
            name = 'omggyy' + str((int(str(self.name)[-1])+self.mother.web_shift) % 16)
            try:
                client_player = holdem.ClientPlayer(uri, name, self.AC, debug=False, playing_live=False)
                client_player.doListen()
            except:
                self.mother.web_shift += self.mother.n_workers

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




    def work(self, opposite_agents, max_global_ep, update_global_iter, gamma, dump_global_iter, uri=None, oppositenum=9):
        self.mother.final_dumped = False
        local_game_count = 0
        local_round_count = 0
        geterror = False
        state = None

        self.env_init(opposite_agents, oppositenum)
        self.learnable_agent = self._get_learnable_agent(self.model_list)

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
                        print('====> repeat_round_card!')
                        break


                    # if self.name == 'W_0':
                    #     self.env.render()
                    current_player = state.community_state.current_player
                    actions = holdem.model_list_action(state, n_seats=self.env.n_seats, model_list=self.model_list)

                    # state, rews, cycle_terminal, info = self.env.step(actions)

                    try:
                        state, rews, cycle_terminal, info = self.env.step(actions)
                    except (ValueError, gym.error.Error, KeyError) as e:
                        print('=====>', e)
                        geterror = True
                        break

                # print('round end')
                state = self.env.reset()

                for s, a in self.learnable_agent.items():
                    a.round_end(state, s)

            for s, a in self.learnable_agent.items():
                a.game_over(state, s)

            local_round_count = 0


    @staticmethod
    def _get_stack(state):
        # print(state.community_state.button)
        # print([i.stack + i.betting for i in state.player_states])
        # print(sum([i.stack + i.betting for i in state.player_states]) + state.community_state.bigblind + state.community_state.smallblind)
        return [i.stack for i in state.player_states]

    @staticmethod
    def _get_v(buffer_r, gamma=0.99):
        v_s_ = 0
        buffer_v_target = list()
        for r in buffer_r[::-1]:  # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        return buffer_v_target

    def get_round_reward(self, next_stack):
        rewards = [i - j for i, j in zip(next_stack, self.round_start_stack)]
        return rewards

    def update_round_start_stack(self, next_stack):
        self.round_start_stack = next_stack

    @staticmethod
    def _get_learnable_agent(model_list):
        return {i: model for i, model in enumerate(model_list) if hasattr(model, 'update_buffer')}

    def update_player_winning_cache(self, rewards):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            for s, pr in enumerate(rewards):
                agent.player_cache[s].update({'winned': 0 if not pr else (pr/abs(pr))})

    def init_agent_player_cache(self):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            agent.player_cache = PLAYER_CACHES_INIT.copy()

    def agents_learning(self):
        # print('learning')
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            if agent.buffer_s:
                buffer_s = np.vstack(agent.buffer_s)
                buffer_action = np.vstack(agent.buffer_action)
                buffer_amount = np.vstack(agent.buffer_amount)
                buffer_v = np.vstack(agent.buffer_v)

                feed_dict = {
                    agent.s: buffer_s,
                    agent.dis_a_his: buffer_action,
                    agent.con_a_his: buffer_amount,
                    agent.v_target: buffer_v,
                }

                agent.update_global(feed_dict)
                agent.pull_global()
                agent.init_learning_buffer()


    def agents_pull(self):
        # print('learning')
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            agent.pull_global()

    def env_init(self, opposite_agents, oppositenum):

        shuffle(opposite_agents)
        self.model_list = opposite_agents.copy()[:oppositenum]
        self.seats = oppositenum + 1
        self.myseat = np.random.randint(self.seats)
        self.model_list.insert(self.myseat, self.AC)
        self.env = gym.make('TexasHoldem-v2')  # holdem.TexasHoldemEnv(self.seats)

        # gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(self.seats)  #  holdem.TexasHoldemEnv(2)
        for i in range(self.seats):
            self.env.add_player(i, stack=3000)
        self.round_start_stack = self._get_stack(self.env.reset())

    def init_agent_cache(self):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            agent.init_cache()

    def update_v_buffer(self, gamma=0.99):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            if not agent.buffer_r:  # for non action round, ex: as play as bb while all others fold
                continue
            buffer_v_target = self._get_v(agent.buffer_r, gamma=gamma)
            agent.buffer_v.extend(buffer_v_target)
            agent.init_r()

    def reward_correction(self, rewards):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            if not agent.buffer_r:  # for non action round, ex: as play as bb while all others fold
                continue
            agent.buffer_r[-1] = rewards[seat]

