import numpy as np
import tensorflow as tf
import itertools
import collections

from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import card

PLAYER_CACHES = ['isallin', 'betting', 'playing_hand', 'count', 'winned']
PLAYER_CACHES_INIT = collections.defaultdict(lambda: collections.Counter(PLAYER_CACHES))

class ACNet:

    def __init__(self, mother, scope, sess, globalmodel=None, a_opt=None, c_opt=None, training=False):
        # self.global_net = 'global_net'
        self.mother = mother
        self.sess = sess
        self.name = scope
        self.state_size = 259
        self.dis_action_space = 3
        self.con_action_space = 1
        self.con_action_bound = [0, 30]

        self.game_round = 0
        self.game_count = 0
        self.ep_r = 0

        self.con_weight = 0.02
        self.gamma = 0.95
        self.con_entropy_beta = 0.01
        self.dis_entropy_beta = 0.05
        self.training = training
        self.globalmodel = globalmodel

        self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_r, self.buffer_v = list(), list(), list(), list(), list()
        # self.state_cache, self.action_cache, self.amount_cache = None, None, None
        self.player_cache = PLAYER_CACHES_INIT.copy()

        self.build_agent(scope=scope, a_opt=a_opt, c_opt=c_opt)

        self.round_start_stacks = None

        if globalmodel is not None:
            print(self.name, 'pulled')
            self.pull_global()

    def build_agent(self, scope, a_opt, c_opt):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False)
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
                    self.con_a_loss = tf.reduce_mean(-self.con_exp_v)

                self.a_loss = self.dis_a_loss + self.con_weight * self.mask * self.con_a_loss

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
        amount = float(amount[0])
        return action, amount

    def self_attention(self):
        pass ## TODO

    def _build_net(self, scope, layer_nodes=256, convergence_node=4):
        # w_init = tf.random_normal_initializer(0, 0.1)
        w_init = tf.glorot_uniform_initializer()
        drop_prob = 0.2
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, layer_nodes, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)
            l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a_ = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_a
            # l_a = tf.layers.dropout(l_a_, rate=drop_prob)
            l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a_ = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_a_
            # l_a = tf.layers.dropout(l_a_, rate=drop_prob)
            l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            # l_a = tf.layers.dropout(l_a, rate=drop_prob)
            # l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_a_


            # l_a_actions = tf.layers.dense(l_a, self.dis_action_space*convergence_node, tf.nn.relu, kernel_initializer=w_init)
            # l_a_mu = tf.layers.dense(l_a, self.con_action_space*convergence_node, tf.nn.relu, kernel_initializer=w_init)
            # l_a_sigma = tf.layers.dense(l_a, self.con_action_space*convergence_node, tf.nn.relu, kernel_initializer=w_init)

            actions = tf.layers.dense(l_a, self.dis_action_space, tf.nn.softmax, kernel_initializer=w_init, name='actions')  # raise, call, check, fold
            mu = tf.layers.dense(l_a, self.con_action_space, tf.nn.relu, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.con_action_space, tf.nn.relu, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, layer_nodes, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            # l_c = tf.layers.dropout(l_c, rate=drop_prob)
            l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            l_c = tf.layers.dropout(l_c, rate=drop_prob)
            # l_c_ = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_c
            # l_c = tf.layers.dropout(l_c_, rate=drop_prob)
            l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            # l_c = tf.layers.dropout(l_c, rate=drop_prob)
            # l_c_ = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_c_
            # l_c = tf.layers.dropout(l_c_, rate=drop_prob)
            l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
            # l_c = tf.layers.dropout(l_c, rate=drop_prob)
            # l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init) + l_c_

            # l_c = tf.layers.dense(l_c, convergence_node, tf.nn.relu, kernel_initializer=w_init)
            v = tf.layers.dense(l_c, 1, tf.nn.relu, kernel_initializer=w_init, name='v')  # state value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return actions, mu, sigma, v, a_params, c_params

    def getReload(self, state):
        # print('we got asking for reloading')
        return False

    # def reset_cache(self):
    #     self.player_cache = PLAYER_CACHES_INIT.copy()
    #     self.init_cache()
    #     self.init_learning_buffer()
    #     self.init_r()

    def new_round(self, state, playerid):
        self.round_start_stacks = [i.stack for i in state.player_states]
        # print('new_round')

    def round_end(self, state, playerid):
        stacks = [i.stack for i in state.player_states]
        rewards = [i - j for i, j in zip(stacks, self.round_start_stacks)]

        self.game_round += 1
        self.ep_r += rewards[playerid]

        for s, pr in enumerate(rewards):
            self.player_cache[s].update({'winned': 0 if not pr else (pr / abs(pr))})

        if self.buffer_r:  # for non action round, ex: as play as bb while all others fold
            r = rewards[playerid] / 1000
            self.buffer_r[-1] = r  # (r / (abs(r)+1)) * (r ** 2)
            buffer_v_target = self._get_v(self.buffer_r, gamma=self.gamma)

            # print(self.name, 'buffer_v_target', buffer_v_target)
            # print(self.name, 'buffer_r', self.buffer_r)

            self.buffer_v.extend(buffer_v_target)
            self.init_r()

            # self.init_cache()

        # print('round_end')

    def learning(self):
        # print(self.name, 'self.buffer_v', self.buffer_v)
        # print(self.name, 'self.buffer_action', self.buffer_action)
        # print(self.name, 'self.buffer_amount', self.buffer_amount)
        # print(self.name, self.buffer_s)
        buffer_s = np.vstack(self.buffer_s)
        buffer_action = np.vstack(self.buffer_action)
        buffer_amount = np.vstack(self.buffer_amount)
        buffer_v = np.vstack(self.buffer_v)
        feed_dict = {
            self.s: buffer_s,
            self.dis_a_his: buffer_action,
            self.con_a_his: buffer_amount,
            self.v_target: buffer_v,
        }
        self.update_global(feed_dict)
        self.pull_global()
        print('learned')

    @staticmethod
    def _get_v(buffer_r, gamma=0.99):
        v_s_ = 0
        buffer_v_target = list()
        for r in buffer_r[::-1]:  # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        return buffer_v_target

    def game_over(self, state, playerid):

        self.mother.global_ep += 1
        global_ep = self.mother.global_ep
        self.game_count += 1
        if not global_ep % self.mother.update_global_iter:
            if self.training:
                if self.buffer_s:
                    self.learning()
            self.init_learning_buffer()

        if not global_ep % self.mother.dump_global_iter:
            if self.training:
                self.mother.dump_sess(global_step=self.globalmodel.global_step)
                print('ckpt dumped')


        self.mother.global_running_r.append(self.ep_r)
        self.mother.global_running_r = self.mother.global_running_r[-1000:]
        print('{}\tEp:{}| GameRound:{} | Ep_r:{}| Ep_r_avg:{}'.format(
            self.name,
            global_ep,
            self.game_round,
            self.ep_r,
            np.mean(self.mother.global_running_r)))
        self.ep_r = 0
        self.game_round = 0
        # print(self.player_cache)
        self.player_cache = PLAYER_CACHES_INIT.copy()

        # print('game_over')

    def takeAction(self, state, playerid):

        to_call = state[1].call_price
        feature = self.state2feature(state, playerseat=playerid)
        action, amount = self.choose_action(feature)
        amount_ = int(amount * 1000)

        hands = [i for i in state.player_states[playerid].hand if i != -1]
        publics = [i for i in state.community_card if i != -1]
        action2action = {i: a for i, a in enumerate(['bet', 'call', 'check', 'fold'])}
        print(action2action[action], amount_, card.deuces2cards(hands), card.deuces2cards(publics))

        # self.state_cache, self.action_cache, self.amount_cache = feature, action, amount
        self.update_buffer(0, feature, action, amount)  # temp reward

        chips = 0
        for i in state[0]:
            if i.seat == playerid:
                chips = i.stack

        if action == 0:
            if chips < to_call:
                return ACTION(action_table.CALL, int(amount_ + to_call))
            return ACTION(action_table.RAISE, int(amount_ + to_call))

        elif action == 1:
            return ACTION(action_table.CALL, int(amount_+to_call))

        return ACTION(action_table.CHECK, int(amount_ + to_call))
        # elif action == 2:
        #     return ACTION(action_table.CHECK, int(amount_+to_call))
        # return ACTION(action_table.FOLD, int(amount_ + to_call))


    def state2feature(self, state, playerseat):

        o_vector = list()
        p_vector = None
        hands = None

        for p in state[0]:
            if p.seat == playerseat:
                hands = p.hand
                p_vector = [p.isallin, p.playedthisround, p.betting / 1000, p.stack / 1000, p.betting / (p.stack + 1)]
                # 5
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
        hands = [i for i in hands if i != -1]
        publics = [i for i in publics if i != -1]
        hand_vector = card.deuces2features(hands + publics)  # 52
        public_vector = card.deuces2features(publics)  # 52

        return np.concatenate([p_vector, hand_vector, public_vector, t_vector, o_vector])  # 259
        # 5 + 52 + 52 + 6 + 144 = 259

    # def init_cache(self):
    #     self.state_cache, self.action_cache, self.amount_cache = None, None, None

    def init_learning_buffer(self):
        # print(list(map(len, [self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_v])))
        self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_v = list(), list(), list(), list()

    def init_r(self):
        self.buffer_r = list()

    def update_buffer(self, r, feature, action, amount):
        self.buffer_s.append(feature)
        self.buffer_r.append(r)
        self.buffer_action.append(action)
        self.buffer_amount.append(amount)