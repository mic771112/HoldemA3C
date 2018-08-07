import numpy as np
import tensorflow as tf
import itertools
import collections

from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import card

PLAYER_CACHES = ['isallin', 'betting', 'playing_hand', 'count', 'winned']
PLAYER_CACHES_INIT = collections.defaultdict(lambda: collections.Counter(PLAYER_CACHES))

class ACNet:

    def __init__(self, mother, scope, sess, globalmodel=None, con_a_opt=None, dis_a_opt=None, c_opt=None, training=False):

        # self.global_net = 'global_net'
        self.mother = mother
        self.sess = sess
        self.name = scope
        # self.state_size = 259
        self.dis_action_space = 2
        self.con_action_space = 1
        self.con_action_bound = [10, 10000]

        self.game_round = 0
        self.game_count = 0
        self.ep_r = 0

        self.con_weight = 1
        self.gamma = 0.90
        self.con_entropy_beta = 1e0
        self.dis_entropy_beta = 1e2
        self.training = training
        self.globalmodel = globalmodel

        self.buffer_s, self.buffer_action, self.buffer_amount, self.buffer_t, self.buffer_v = list(), list(), list(), list(), list()
        # self.state_cache, self.action_cache, self.amount_cache = None, None, None
        self.player_cache = PLAYER_CACHES_INIT.copy()

        self.build_agent(scope=scope, con_a_opt=con_a_opt, dis_a_opt=dis_a_opt, c_opt=c_opt)

        self.round_start_stacks = None

        # if globalmodel is not None:
        #     print(self.name, 'pulled')
        #     self.pull_global()

    @staticmethod
    def nan_filtering(grads, variables):
        return [(g, v) for g, v in zip(grads, variables) if g is not None]

    def build_agent(self, scope, con_a_opt, dis_a_opt, c_opt):


        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # c_array, o_array, pt_vector = s  ## (14, 4, 3), (16 x n_opposite), (11)

            # self.c_array = tf.placeholder(tf.float32, [None, 14, 4, 3], 'c_array')
            # self.o_array = tf.placeholder(tf.float32, [None, None, 68], 'o_array')
            # self.pt_vector = tf.placeholder(tf.float32, [None, 11], 'pt_vector')

            self.state_vector = tf.placeholder(tf.float32, [None, 992], 's_vector')
            self.state_array = tf.placeholder(tf.float32, [None, None, 81], 's_array')
            self.state_round = tf.placeholder(tf.int32, [None], 'round')

            self.dis_a_his = tf.placeholder(tf.int32, [None, 1], 'Ad')
            self.con_a_his = tf.placeholder(tf.float32, [None, self.con_action_space], 'Ac')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            self.a_prob, self.mu, self.sigma, self.v, self.a_params, self.c_params, self.params = self._build_net(scope)
            self.mask = tf.cast(tf.not_equal(tf.argmax(self.a_prob), 2), tf.float32)  # 2: 0 else 1

            # self.a_prob : [raise, call, check/fold]
            # self.mask = tf.cast(tf.equal(tf.argmax(self.a_prob), 0), tf.float32)  # 0raise:1, else: 0
            normal_dist = tf.distributions.Normal(self.mu, tf.abs(self.sigma))

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
                    self.con_a_loss = self.mask * self.con_a_loss

            with tf.name_scope('choose_amount'):  # use local params to choose action
                value = tf.squeeze(normal_dist.sample(1), axis=[0, 1])
                self.amount = tf.clip_by_value(value**3+1000,
                                               self.con_action_bound[0],
                                               self.con_action_bound[1])

            with tf.name_scope('local_grad'):
                self.con_a_grads = tf.gradients(self.con_a_loss, self.params)
                self.dis_a_grads = tf.gradients(self.dis_a_loss, self.params)
                self.c_grads = tf.gradients(self.c_loss, self.params)
                # self.a_grads = tf.gradients(self.a_loss, self.a_params)
                # self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.globalmodel is not None:

                with tf.name_scope('sync'):

                    with tf.name_scope('pull'):
                        self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in
                                                 zip(self.params, self.globalmodel.params)]
                        # self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in
                        #                          zip(self.a_params, self.globalmodel.a_params)]
                        # self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                        #                          zip(self.c_params, self.globalmodel.c_params)]

                    with tf.name_scope('push'):
                        self.update_con_a_op = con_a_opt.apply_gradients(self.nan_filtering(self.con_a_grads, self.globalmodel.params))
                        self.update_dis_a_op = dis_a_opt.apply_gradients(self.nan_filtering(self.dis_a_grads, self.globalmodel.params))
                        self.update_c_op = c_opt.apply_gradients(self.nan_filtering(self.c_grads, self.globalmodel.params))
                        # self.update_a_op = a_opt.apply_gradients(zip(self.a_grads, self.globalmodel.a_params))
                        # self.update_c_op = c_opt.apply_gradients(zip(self.c_grads, self.globalmodel.c_params))

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_con_a_op, self.update_dis_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run(self.pull_params_op)
        # print(self.sess.run(self.params))
        # print('\n'*10)
        # print(self.sess.run(self.globalmodel.params))
        # self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        # c_array, o_array, pt_vector = s  ## (14, 4, 3), (n_opposite x 16), (11)
        state_vector, state_array, state_round = s

        state_vector = state_vector[np.newaxis, :]
        state_array = state_array[np.newaxis, :]
        state_round = np.array([state_round])

        prob_weights, amount = self.sess.run([self.a_prob, self.amount], {self.state_vector: state_vector,
                                                                          self.state_array: state_array,
                                                                          self.state_round: state_round})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        amount = float(amount[0])
        return action, amount

    @staticmethod
    def build_branches(o_vector, layer_nodes, w_init, drop_prob):
        main_layer = tf.layers.dense(o_vector, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
        main_layer = tf.layers.dropout(main_layer, rate=drop_prob)
        main_layer = tf.layers.dense(main_layer, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
        main_layer = tf.layers.dropout(main_layer, rate=drop_prob)
        main_layer = tf.layers.dense(main_layer, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
        return main_layer


    def _build_net(self, scope, layer_nodes=512, convergence_node=32):
        # w_init = tf.random_normal_initializer(0, 0.1)
        w_init = tf.glorot_uniform_initializer()
        drop_prob = 0.1

        with tf.variable_scope('ac'):
            with tf.variable_scope('main'):
                self.global_step = tf.Variable(0, trainable=True)

                cell_fw = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(i) for i in [64, 64, 64]])
                cell_bw = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(i) for i in [64, 64, 64]])
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.state_array, dtype=tf.float32)
                fw_vector = outputs[0][:, -1, :]
                bw__vector = outputs[1][:, -1, :]
                o_vector = tf.concat([fw_vector, bw__vector, self.state_vector], -1)

                main_layer_0 = self.build_branches(o_vector, layer_nodes, w_init, drop_prob)
                main_layer_3 = self.build_branches(o_vector, layer_nodes, w_init, drop_prob)
                main_layer_4 = self.build_branches(o_vector, layer_nodes, w_init, drop_prob)
                main_layer_5 = self.build_branches(o_vector, layer_nodes, w_init, drop_prob)
                tf.transpose([tf.equal(self.state_round, 0)])
                main_layer = tf.cast(tf.transpose([tf.equal(self.state_round, 0)]), tf.float32) * main_layer_0 \
                             + tf.cast(tf.transpose([tf.equal(self.state_round, 3)]), tf.float32) * main_layer_3 \
                             + tf.cast(tf.transpose([tf.equal(self.state_round, 4)]), tf.float32) * main_layer_4 \
                             + tf.cast(tf.transpose([tf.equal(self.state_round, 5)]), tf.float32) * main_layer_5

            with tf.variable_scope('actor'):
                l_a = tf.layers.dropout(main_layer, rate=drop_prob)
                l_a = tf.layers.dense(l_a, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
                l_a_actions = tf.layers.dense(l_a, self.dis_action_space*convergence_node, tf.nn.relu6, kernel_initializer=w_init)
                l_a_mu = tf.layers.dense(l_a, self.con_action_space*convergence_node, tf.nn.relu6, kernel_initializer=w_init)
                l_a_sigma = tf.layers.dense(l_a, self.con_action_space*convergence_node, tf.nn.relu6, kernel_initializer=w_init)
                actions = tf.layers.dense(l_a_actions, self.dis_action_space, tf.nn.softmax, kernel_initializer=w_init, name='actions')  # raise, call, check, fold
                mu = tf.layers.dense(l_a_mu, self.con_action_space, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a_sigma, self.con_action_space, kernel_initializer=w_init, name='sigma')

            with tf.variable_scope('critic'):
                l_c = tf.layers.dropout(main_layer, rate=drop_prob)
                l_c = tf.layers.dense(l_c, layer_nodes, tf.nn.relu6, kernel_initializer=w_init)
                l_c = tf.layers.dense(l_c, convergence_node, tf.nn.relu6, kernel_initializer=w_init)
                l_c = tf.layers.dense(l_c, convergence_node, tf.nn.relu6, kernel_initializer=w_init)
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ac')
        main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ac/main')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ac/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ac/critic')
        return actions, mu, sigma, v, a_params, c_params, params

    def getReload(self, state):
        # print('we got asking for reloading')
        return True

    # def reset_cache(self):
    #     self.player_cache = PLAYER_CACHES_INIT.copy()
    #     self.init_cache()
    #     self.init_learning_buffer()
    #     self.init_r()

    def new_round(self, state, playerid):
        self.round_start_stacks = [i.stack for i in state.player_states]
        # print('new_round')

    def round_end(self, state, playerid):
        sb = state[1].smallblind

        stacks = [i.stack for i in state.player_states]
        n_players = len(list(filter(lambda x: x > 0, stacks)))
        n_players = n_players if n_players else 1
        turn_reward = {0: 0, 1: 0, 2: 0, 3: 1.1, 4: 1.1, 5: 1.1}
        rewards = [i - j - (3 * sb / n_players) for i, j in zip(stacks, self.round_start_stacks)]

        self.game_round += 1
        self.ep_r += rewards[playerid]

        for s, pr in enumerate(rewards):
            self.player_cache[s].update({'winned': 0 if ((not pr) or (pr<=0)) else 1})

        if self.buffer_t:  # for non action round, ex: as play as bb while all others fold
            turn = self.buffer_t[-1]  # len(list(filter(lambda x: x > 0, state[2])))
            print(self.name, 'turn', turn)

            print('{} ---> round turn: {}\tr: {} <---'.format(self.name, turn, rewards[playerid]), end=' ')
            rewards = [i + (turn_reward[turn] * sb) for i in rewards]  # reward warpping
            # rewards = [i if (i >= 0) else i for i in rewards]  # reward warpping
            # r = rewards[playerid]
            r = rewards[playerid] / 1000
            if turn == 0:  # reward warpping
                r = r/1.1 if (r > 0) else (1.1 * r)
            else:  # reward warpping
                r = r*1.3 if (r > 0) else r

            print('warpped: {}'.format(r))
            buffer_r = [0] * len(self.buffer_t)
            buffer_r[-1] = r  # (r / (abs(r)+1)) * (r ** 2)
            buffer_v_target = self._get_v(buffer_r, gamma=self.gamma)

            # print(self.name, 'buffer_v_target', buffer_v_target)
            # print(self.name, 'buffer_r', self.buffer_t)

            self.buffer_v.extend(buffer_v_target)
            self.init_r()

            # self.init_cache()

        # print('round_end')

    def learning(self):

        state_vector, state_array, state_round = zip(*self.buffer_s)
        state_vector = np.array(state_vector)
        max_opposite = max([i.shape[0] for i in state_array])
        state_array = np.array([np.vstack([i, [[0] * 81] * (max_opposite - i.shape[0])])
                                if (max_opposite - i.shape[0]) else i for i in state_array])
        state_round = np.array(state_round)
        buffer_action = np.vstack(self.buffer_action)
        buffer_amount = np.vstack(self.buffer_amount)
        buffer_v = np.vstack(self.buffer_v)
        feed_dict = {
            self.state_vector: state_vector,
            self.state_array: state_array,
            self.state_round: state_round,
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

    def game_start(self, state, playerid):
        self.ep_r = 0

    def game_over(self, state, playerid):
        print('game_over', 'game_over')

        try:
            self.mother.global_ep += 1
            global_ep = self.mother.global_ep
            self.game_count += 1
            if not self.game_count % self.mother.update_iter:
                if self.training:
                    if self.buffer_s:
                        self.learning()
                self.init_learning_buffer()
            if not global_ep % self.mother.dump_global_iter:
                if self.training:
                    self.mother.dump_sess()
                    print('ckpt dumped')
            print('self.mother.global_ep', self.mother.global_ep)
            print('global_ep', global_ep, '/', self.mother.dump_global_iter)
        except Exception as e:
            print(e)
            print(e)
            print(e)
            print(e)



        self.mother.global_running_r.append(self.ep_r)
        self.mother.global_running_r = self.mother.global_running_r[-1000:]
        print('{} =======>\tEp:{}| GameRound:{} | Ep_r:{}| Ep_r_avg:{} <========'.format(
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

        to_call = state[1].to_call
        # to_call = state[1].call_price

        # print('state[1].call_price', state[1].call_price)
        # print('state[1].to_call', state[1].to_call)

        featurearrays = self.state2featurearrays(state, playerseat=playerid) #
        action, amount = self.choose_action(featurearrays)

        amount_ = int(amount)  #  * 1000

        hands = [i for i in state.player_states[playerid].hand if i != -1]
        publics = [i for i in state.community_card if i != -1]
        action2action = {i: a for i, a in enumerate(['bet', 'call', 'check', 'fold'])}

        # self.state_cache, self.action_cache, self.amount_cache = feature, action, amount
        self.update_buffer(len(publics), featurearrays, action, amount)  # temp reward
        chips = 0
        for i in state[0]:
            if i.seat == playerid:
                chips = i.stack

        # if action == 0:
            # # if chips < to_call:
            # if (chips < to_call) or (amount_ < to_call):
            #     print(self.name, 'action:', action, 'CALL', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
            #     # return ACTION(action_table.CALL, int(amount_ + to_call))
            #     return ACTION(action_table.CALL, int(amount_))
            # print(self.name, 'action:', action, 'RAISE', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
            # # return ACTION(action_table.RAISE, int(amount_ + to_call))
            # return ACTION(action_table.RAISE, int(amount_))

        if (action == 0) and (amount_ >= to_call) and (chips >= to_call):
            print(self.name, 'chips', chips, 'action:', action, 'RAISE', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
            return ACTION(action_table.RAISE, int(amount_))

        # elif action == 1:
        if amount_ >= to_call:
            print(self.name, 'chips', chips, 'action:', action, 'CALL', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
            return ACTION(action_table.CALL, int(to_call))

        if to_call > 0:
            print(self.name, 'chips', chips, 'action:', action, 'FOLD', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
            return ACTION(action_table.FOLD, int(0))
        print(self.name, 'chips', chips, 'action:', action, 'CHECK', amount_, to_call, card.deuces2cards(hands), card.deuces2cards(publics))
        return ACTION(action_table.CHECK, int(0))
        # elif action == 2:
        #     return ACTION(action_table.CHECK, int(amount_+to_call))
        # return ACTION(action_table.FOLD, int(amount_ + to_call))

    @staticmethod
    def numerical_transform(vector):  # n
        try:
            assert all([i>=0 for i in vector])
            transformed_vector = vector.copy()
            transformed_vector += list(map(lambda x: 1 / max(1, x + 1), vector))
            transformed_vector += list(map(np.square, vector))
            transformed_vector += list(map(lambda x: np.log1p(max(0, x + 1)), vector))
            transformed_vector += list(map(lambda x: np.square(1 / max(1, x + 1)), vector))
            transformed_vector += list(map(lambda x: np.log1p(1 / max(1, x + 1)), vector))

        except:
            print('error!!!')
            print(vector)
            return
        return transformed_vector  # n x 6

    @staticmethod
    def np_numerical_transform(vector):  # n
        vector_ = np.array(vector)
        invert = 1 / np.maximum(1, vector_ + 1)
        transformed_vector = np.concatenate([vector_,
                                             invert,
                                             np.square(vector_),
                                             np.log1p(vector_),
                                             np.square(invert),
                                             np.log1p(invert)])
        return transformed_vector  # n x 6

    def state2featurearrays(self, state, playerseat):

        o_vector = list()
        p_vector = None
        hands = None

        for p in state[0]:
            if p.seat == playerseat:
                hands = p.hand
                p_state_vector = np.array([p.isallin, p.playedthisround]) # 2
                p_count_vector = np.array([p.betting / 1000, p.stack / 1000, p.betting / (p.stack + 100)]) # 3

                p_vector = np.concatenate([p_state_vector, self.np_numerical_transform(p_count_vector)])  # 2 + 3*6 = 20
                continue

            self.player_cache[p.seat].update({'isallin': int(p.isallin > 0),
                                              'betting': int(p.betting > 0),
                                              'playing_hand': int(p.playing_hand > 0),
                                              'count': 1,
                                              'winned': 0,
                                              })
            if not p.playing_hand:
                continue

            o_state = np.array([int(p.isallin > 0), int(p.playedthisround > 0), int(p.betting > 0)]) # 3

            p_cache = self.player_cache[p.seat]
            o_counts = [max(0, p.betting+1) / 1000,
                        max(0, p.stack+1) / 1000,
                        max(0, p.betting+1) / max(1, p.stack+1),
                        p_cache['isallin']+1 / p_cache['count'],
                        p_cache['betting'] / p_cache['count'],
                        p_cache['playing_hand'] / p_cache['count'],
                        p_cache['winned'] / p_cache['count'],
                        p_cache['isallin'] / p_cache['playing_hand'],
                        p_cache['betting'] / p_cache['playing_hand'],
                        p_cache['winned'] / p_cache['playing_hand'],
                        p_cache['winned'] / p_cache['isallin'],
                        p_cache['winned'] / p_cache['betting'],
                        p_cache['winned'] / p_cache['playing_hand']]  # 13

            o_vector.append(np.concatenate([o_state, self.numerical_transform(o_counts)])) # 3 + 13*6 = 81
        if not o_vector:
            o_vector.append([0] * 81)
        state_array = np.array(o_vector)  # n_opposite x 81

        if p_vector is None:
            p_vector = [0] * 20
            p_vector = np.array(p_vector)  # 20

        table = state[1]

        t_count_vector = [table.smallblind / 100,
                          table.totalpot / 1000,
                          table.lastraise / 1000,
                          table.call_price / 1000,
                          table.to_call / 1000,
                          state_array.shape[0] / 10]  # 6

        t_vector = self.np_numerical_transform(t_count_vector)  # 6 * 6 = 36

        pt_vector = np.concatenate([p_vector, t_vector])  # 20 + 36 = 56

        publics = state[2]
        hands = [i for i in hands if i != -1]
        publics = [i for i in publics if i != -1]

        hand_array = self.np_numerical_transform(card.deuces2features(hands))  # 14x4x6 = 312
        public_array = self.np_numerical_transform(card.deuces2features(publics))  # 14x4x6 = 312
        card_array = self.np_numerical_transform(card.deuces2features(hands + publics))  # 14x4x6 = 312
        state_vector = np.concatenate([card_array, public_array, hand_array, pt_vector])

        return state_vector, state_array, len(publics)  ## (312*3 + 56 = 992), (n_opposite x 81), turn_number

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
        self.buffer_t = list()

    def update_buffer(self, r, feature, action, amount):
        self.buffer_s.append(feature)
        self.buffer_t.append(r)
        self.buffer_action.append(action)
        self.buffer_amount.append(amount)