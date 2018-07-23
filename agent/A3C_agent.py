from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import gym
import holdem
import agent

import tensorflow as tf
import numpy as np
import threading
import multiprocessing
import os
import shutil
import itertools
import time
import pathlib
import card
import collections

import logging

logging.basicConfig(level=logging.DEBUG)


PLAYER_CACHES = ['isallin', 'betting', 'playing_hand', 'count', 'winned']
PLAYER_CACHES_INIT = collections.defaultdict(lambda: collections.Counter(PLAYER_CACHES))

class A3CAgent:

    def __init__(self, model_dir, learning=True):

        self.sess = None
        self.saver = None
        self.workers = None
        self.learning = learning
        self.global_scope = 'global_net'
        self.output_graph = True

        self.model_dir = model_dir
        self.log_dir = '{}/log'.format(self.model_dir)
        self.ckpt_path = '{}/ckpt'.format(self.model_dir)
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.lr_a = 0.00001
        self.lr_c = 0.00001

        self.n_workers = multiprocessing.cpu_count()
        self.global_net = ACNet(scope=self.global_scope,
                                sess=self.sess,
                                globalmodel=None,
                                a_opt=None,
                                c_opt=None)
        self.load_sess()
        self.global_net.sess = self.sess

        self.opt_a = tf.train.RMSPropOptimizer(self.lr_a, name='RMSPA')
        self.opt_c = tf.train.RMSPropOptimizer(self.lr_c, name='RMSPC')

        self.coord = tf.train.Coordinator()
        self.output_graph = True

        self.global_running_r = list()
        self.global_ep = 0


        self.hiring()

    def dump_tensorboard(self):
        if self.output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir, ignore_errors=True)
            tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def load_sess(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(name='saver')

        try:
            self.saver.restore(self.sess, self.ckpt_path)
            logging.error('ckpt read')

        except tf.errors.NotFoundError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            logging.error('no ckpt read')


    def dump_sess(self, global_step=None, extra_path=''):
        path = self.ckpt_path + extra_path

        try:
            if global_step is not None:
                self.saver.save(self.sess, path, global_step=global_step)
            else:
                self.saver.save(self.sess, path)
            logging.error('ckpt saved: {}'.format(path))
        except:
            time.sleep(5)
            self.dump_sess(global_step=global_step, extra_path=extra_path)

    def hiring(self):

        with tf.device("/cpu:0"):
            self.workers = list()
            # Create worker
            for i in range(self.n_workers):
                i_name = 'W_%i' % i  # worker name
                self.workers.append(Worker(name=i_name,
                                           sess=self.sess,
                                           globalmodel=self.global_net,
                                           a_opt=self.opt_a,
                                           c_opt=self.opt_c,
                                           learning=self.learning
                                           ))
            self.sess.run(tf.global_variables_initializer())
            self.load_sess()
            for worker in self.workers:
                worker.AC.pull_global()

    def train(self, opposite_agents, max_global_ep=100, update_global_iter=137, gamma=0.99, dump_global_iter=3571):

        self.dump_tensorboard()
        worker_threads = list()
        for worker in self.workers:

            def job():
                try:
                    worker.work(opposite_agents=opposite_agents.copy(),
                                mother=self,
                                max_global_ep=max_global_ep,
                                update_global_iter=update_global_iter,
                                gamma=gamma,
                                dump_global_iter=dump_global_iter)
                except ValueError as e:
                    while True:
                        print(e)
                        self.sess.run(worker.AC.pull_global())
                        worker.work(opposite_agents=opposite_agents.copy(),
                                    mother=self,
                                    max_global_ep=max_global_ep,
                                    update_global_iter=update_global_iter,
                                    gamma=gamma,
                                    dump_global_iter=dump_global_iter)
                # worker.work(opposite_agents=opposite_agents.copy(),
                #             mother=self,
                #             max_global_ep=max_global_ep,
                #             update_global_iter=update_global_iter,
                #             gamma=gamma)
            # t = multiprocessing.Process(target=job)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        self.coord.join(worker_threads)

    def getReload(self, state):
        return self.global_net.getReload(state)

    def takeAction(self, state, playerid):
        return self.global_net.takeAction(state, playerid)




class ACNet:

    def __init__(self, scope, sess, globalmodel=None, a_opt=None, c_opt=None, training=False):
        # self.global_net = 'global_net'
        self.sess = sess
        self.state_size = 259
        self.dis_action_space = 4
        self.con_action_space = 1
        self.con_action_bound = [0, 30000]

        self.con_entropy_beta = 0.0001
        self.dis_entropy_beta = 0.001

        self.training = training
        self.globalmodel = globalmodel

        self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_r, self.buffer_v = list(), list(), list(), list(), list()
        self.state_cache, self.action_cache, self.amount_cache = None, None, None
        self.player_cache = PLAYER_CACHES_INIT.copy()

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            self.s = tf.placeholder(tf.float32, [None, self.state_size], 'S')
            self.dis_a_his = tf.placeholder(tf.int32, [None, 1], 'Ad')
            self.con_a_his = tf.placeholder(tf.float32, [None, self.con_action_space], 'Ac')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            self.a_prob, self.mu, self.sigma, self.v, self.a_params, self.c_params = self._build_net(scope)
            # self.a_prob : [raise, call, check, fold]
            self.mask = tf.cast(tf.not_equal(tf.argmax(self.a_prob), 0), tf.float32)  # con loss mask for non-raise

            normal_dist = tf.distributions.Normal(self.mu, self.sigma + 1e-5)

            td = tf.subtract(self.v_target, self.v, name='TD_error')
            with tf.name_scope('c_loss'):
                self.c_loss = tf.reduce_mean(tf.square(td))

            with tf.name_scope('a_loss'):
                with tf.name_scope('dis_loss'):
                    self.dis_log_prob = tf.reduce_sum(
                        tf.log(self.a_prob + 1e-5) * tf.one_hot(self.dis_a_his, self.dis_action_space, dtype=tf.float32),
                        axis=1, keepdims=True)
                    self.dis_exp_v = self.dis_log_prob * tf.stop_gradient(td)
                    self.dis_entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keepdims=True)
                    self.dis_exp_v = self.dis_entropy_beta * self.dis_entropy + self.dis_exp_v
                    self.dis_a_loss = tf.reduce_mean(-self.dis_exp_v)

                with tf.name_scope('con_loss'):
                    self.con_log_prob = normal_dist.log_prob(self.con_a_his)
                    self.con_exp_v = self.con_log_prob * tf.stop_gradient(td)
                    self.con_entropy = normal_dist.entropy()  # encourage exploration
                    self.con_exp_v = self.con_entropy_beta * self.con_entropy + self.con_exp_v
                    self.con_a_loss = tf.reduce_mean(-self.con_exp_v) / 500

                self.a_loss = self.dis_a_loss + self.mask * self.con_a_loss

            with tf.name_scope('choose_amount'):  # use local params to choose action
                self.amount = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]),
                                               self.con_action_bound[0],
                                               self.con_action_bound[1])

            with tf.name_scope('local_grad'):
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.globalmodel is not None:

                with tf.name_scope('sync'):

                    with tf.name_scope('pull'):
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.a_params, self.globalmodel.a_params)]
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.c_params, self.globalmodel.c_params)]

                    with tf.name_scope('push'):
                        self.update_a_op = a_opt.apply_gradients(zip(self.a_grads, self.globalmodel.a_params))
                        self.update_c_op = c_opt.apply_gradients(zip(self.c_grads, self.globalmodel.c_params))

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]  # np.newaxis: extra dim
        prob_weights, amount = self.sess.run([self.a_prob, self.amount], {self.s: s})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        amount = int(amount[0])
        return action, amount

    def _build_net(self, scope, layer_nodes=256):
        # w_init = tf.random_normal_initializer(0, 0.1)
        w_init = tf.glorot_uniform_initializer()
        drop_prob = 0.3
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, layer_nodes, tf.nn.relu, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dropout(l_a, rate=drop_prob)
            l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)

            l_a_actions = tf.layers.dense(l_a, self.dis_action_space*4, tf.nn.relu, kernel_initializer=w_init)
            l_a_mu = tf.layers.dense(l_a, self.con_action_space*4, tf.nn.relu, kernel_initializer=w_init)
            l_a_sigma = tf.layers.dense(l_a, self.con_action_space*4, tf.nn.relu, kernel_initializer=w_init)

            actions = tf.layers.dense(l_a_actions, self.dis_action_space, tf.nn.softmax, kernel_initializer=w_init, name='actions')  # raise, call, check, fold
            mu = tf.layers.dense(l_a_mu, self.con_action_space, tf.nn.relu, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a_sigma, self.con_action_space, tf.nn.relu, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, layer_nodes, tf.nn.relu, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dropout(l_c, rate=drop_prob)
            l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            l_c = tf.layers.dropout(l_c, rate=drop_prob)
            # l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            # l_c = tf.layers.dropout(l_c, rate=drop_prob)
            # l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu, kernel_initializer=w_init)
            # l_c = tf.layers.dropout(l_c, rate=drop_prob)
            l_c = tf.layers.dense(l_c, 4, tf.nn.relu, kernel_initializer=w_init)

            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return actions, mu, sigma, v, a_params, c_params

    def getReload(self, state):
        return False

    def takeAction(self, state, playerid):
        to_call = state[1].call_price
        feature = self.state2feature(state, playerseat=playerid)
        action, amount = self.choose_action(feature)

        self.state_cache, self.action_cache, self.amount_cache = feature, action, amount

        chips = 0
        for i in state[0]:
            if i.seat == playerid:
                chips = i.stack

        if action == 0:
            if chips < to_call:
                return ACTION(action_table.CALL, int(amount + to_call))
            return ACTION(action_table.RAISE, int(amount + to_call))

        elif action == 1:
            return ACTION(action_table.CALL, int(amount+to_call))

        elif action == 2:
            return ACTION(action_table.CHECK, int(amount+to_call))
        return ACTION(action_table.FOLD, int(amount + to_call))


    def state2feature(self, state, playerseat):

        o_vector = list()
        p_vector = None
        hands = None

        for p in state[0]:
            if p.seat == playerseat:
                hands = p.hand
                p_vector = [p.isallin, p.playedthisround, p.betting / 1000, p.stack / 1000, p.betting / (p.stack + 1)]
                continue
            if not p.playing_hand:
                continue
            self.player_cache[p.seat].update({'isallin': int(p.isallin),
                                              'betting': int(p.betting > 0),
                                              'playing_hand': int(p.playing_hand),
                                              'count': 1,
                                              'winned': 0,
                                              })
            p_cache = self.player_cache[p.seat]
            o_vector.append([int(p.isallin),
                             int(p.playedthisround),
                             int(p.betting > 0),
                             p.betting / 1000,
                             p.stack / 1000,
                             p.betting / (p.stack + 1),
                             p_cache['isallin'] / p_cache['count'],
                             p_cache['betting'] / p_cache['count'],
                             p_cache['playing_hand'] / p_cache['count'],
                             p_cache['winned'] / p_cache['count'],

                             p_cache['isallin'] / p_cache['playing_hand'],
                             p_cache['betting'] / p_cache['playing_hand'],
                             p_cache['winned'] / p_cache['playing_hand'],

                             p_cache['winned'] / p_cache['isallin'],
                             p_cache['winned'] / p_cache['betting'],
                             p_cache['winned'] / p_cache['playing_hand'],
                             ]) # 16

        if not o_vector:
            o_vector.append([0] * 16)
        o_temp = list()
        for i, v in enumerate(itertools.cycle(o_vector)):
            if i >= 9:
                break
            o_temp.append(v)
        o_vector = np.concatenate(o_temp)  # 16*9 = 144

        if p_vector is None:
            p_vector = [0] * 5
        p_vector = np.array(p_vector)  # 5

        table = state[1]
        t_vector = np.array([table.smallblind / 100,
                             table.totalpot / 1000,
                             table.lastraise / 1000,
                             table.call_price / 1000,
                             table.to_call / 1000,
                             table.current_player / 10])  # 6

        publics = state[2]

        hand_vector = card.deuces2features((i for i in hands if i != -1))
        public_vector = card.deuces2features((i for i in publics if i != -1))

        return np.concatenate([p_vector, hand_vector, public_vector, t_vector, o_vector])  # 259


    def init_cache(self):
        self.state_cache, self.action_cache, self.amount_cache = None, None, None

    def init_learning_buffer(self):
        self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_v = list(), list(), list(), list()

    def init_r(self):
        self.buffer_r = list()

    def update_buffer(self, r):
        self.buffer_s.append(self.state_cache)
        self.buffer_r.append(r)
        self.buffer_action.append(self.action_cache)
        self.buffer_amount.append(self.amount_cache)


class WebWorker:
    def __init__(self, name, sess, globalmodel, a_opt, c_opt, learning=True):
        self.sess = sess
        self.name = name
        self.AC = ACNet(name, sess, globalmodel, a_opt, c_opt, training=True)

    # def work(self, opposite_agents, mother, max_global_ep, update_global_iter, gamma, dump_global_iter)
    #     while not mother.coord.should_stop() and mother.global_ep < max_global_ep:  # single move in this loop is a game == a episode
    #         pass

class Worker:
    def __init__(self, name, sess, globalmodel, a_opt, c_opt, learning=True):
        self.sess = sess
        self.name = name
        self.AC = ACNet(name, sess, globalmodel, a_opt, c_opt, training=True)
        self.learning = learning
        self.myseat = None
        self.model_list = None
        self.seats = None
        self.env = None
        self.round_start_stack = None

    def work(self, opposite_agents, mother, max_global_ep, update_global_iter, gamma, dump_global_iter):

        local_game_count = 0
        local_round_count = 0
        geterror = False

        self.env_init(opposite_agents)
        self.agents_pull()

        while not mother.coord.should_stop() and mother.global_ep < max_global_ep:  # single move in this loop is a game == a episode

            mother.global_ep += 1
            local_game_count += 1

            global_ep = mother.global_ep
            self.env_init(opposite_agents)
            self.init_agent_player_cache()

            if geterror:
                for seat, agnet in self._get_learnable_agent(self.model_list).items():
                    agnet.init_cache()
                    agnet.init_learning_buffer()
                    agnet.init_r()
            geterror = False

            ep_r = 0
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
                assert not cycle_terminal
                while not cycle_terminal:  # single move in this loop is a action

                    # print('action start')

                    # if self.name == 'W_0':
                    #     self.env.render()
                    current_player = state.community_state.current_player
                    actions = holdem.model_list_action(state, n_seats=self.env.n_seats, model_list=self.model_list)

                    # state, rews, cycle_terminal, info = self.env.step(actions)
                    try:
                        state, rews, cycle_terminal, info = self.env.step(actions)
                    except (ValueError, gym.error.Error, KeyError):
                        geterror = True
                        break

                    if hasattr(self.model_list[current_player], 'update_buffer'):
                        self.model_list[current_player].update_buffer(
                            0)  # non 0 happened only when cycle_terminal, which will corrected later

                state = self.env.reset()
                next_stack = self._get_stack(state)
                rewards = self.get_round_reward(next_stack)
                ep_r += rewards[self.myseat]
                rewards = [i / 1000 for i in rewards]
                self.update_round_start_stack(next_stack)
                self.update_player_winning_cache(rewards)

                self.reward_correction(rewards)  # correct the round end reward
                self.update_v_buffer(gamma=gamma)  # update v and clear r
                self.init_agent_cache()

            # print('game over')
            if self.learning:
                if not global_ep % update_global_iter:
                    self.agents_learning()  # learn and clear v
                    print('learned!')

                if not global_ep % dump_global_iter:
                    mother.dump_sess()  # learn and clear v
                    # print('ckpt dumped!')


            local_round_count = 0

            mother.global_running_r.append(ep_r)
            mother.global_running_r = mother.global_running_r[-100:]
            print('{}\tEp:{}| GameRound:{} | Ep_r:{}| Ep_r_avg:{}'.format(
                self.name,
                global_ep,
                game_round,
                ep_r,
                np.mean(mother.global_running_r)))

        # final dump
        print('final dump')
        mother.dump_sess()

    @staticmethod
    def _get_stack(state):
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
            pr = rewards[seat]
            agent.player_cache[seat].update({'winned': 0 if not pr else (pr/abs(pr))})

    def init_agent_player_cache(self):
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():
            agent.player_cache = PLAYER_CACHES_INIT.copy()
    def agents_learning(self):
        # print('learning')
        learnableagents = self._get_learnable_agent(self.model_list)
        for seat, agent in learnableagents.items():

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

    def env_init(self, opposite_agents):
        from random import shuffle
        shuffle(opposite_agents)
        self.seats = len(opposite_agents) + 1
        self.myseat = np.random.randint(self.seats)

        self.model_list = opposite_agents.copy()
        self.model_list.insert(self.myseat, self.AC)

        self.env = holdem.TexasHoldemEnv(self.seats)

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



if __name__ == '__main__':
    from agent import allinModel
    from agent import allFoldModel
    from agent import allRaiseModel
    from agent import allCallModel

    agent = A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/modoel14', learning=True)
    # agent = A3CAgent(model_dir='./model')
    o_list = [allCallModel()] * 8 \
             + [allRaiseModel()]
    #          # + [allFoldModel()] * 1 \
    #          # + [allRaiseModel()] * 1

    # o_list = [A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/modoel14', learning=True),
    #           A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/modoel14', learning=True)]

    agent.train(opposite_agents=o_list, max_global_ep=1000, dump_global_iter=100, update_global_iter=int(np.exp(0)))  # 3571
    agent.train(opposite_agents=o_list, max_global_ep=3000, dump_global_iter=127, update_global_iter=int(np.exp(1)))
    agent.train(opposite_agents=o_list, max_global_ep=7000, dump_global_iter=3571, update_global_iter=int(np.exp(2)))
    agent.train(opposite_agents=o_list, max_global_ep=10000, dump_global_iter=3571, update_global_iter=int(np.exp(3)))
    agent.train(opposite_agents=o_list, max_global_ep=10000, dump_global_iter=3571, update_global_iter=int(np.exp(4)))
    agent.train(opposite_agents=o_list, max_global_ep=100000, dump_global_iter=3571, update_global_iter=int(np.exp(5)))
