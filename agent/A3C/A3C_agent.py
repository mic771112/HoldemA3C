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

from Workers import Worker
from ACNet import ACNet

logging.basicConfig(level=logging.DEBUG)

class A3CAgent:

    def __init__(self, model_dir, learning=True, hiring=True, n_workers=None, update_iter=10, dump_global_iter=100):

        self.sess = None
        self.saver = None
        self.workers = None
        self.learning = learning
        self.global_scope = 'global_net'
        self.output_graph = True

        self.model_dir = model_dir
        self.log_dir = '{}/log/'.format(self.model_dir)
        self.ckpt_path = '{}/ckpt/'.format(self.model_dir)

        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.lr_a_dis = 1e-5
        self.lr_a_con = 1e-5
        self.lr_c = 1e-5

        self.update_iter = update_iter
        self.dump_global_iter = dump_global_iter
        self.final_dumped = False
        if n_workers is None:
            self.n_workers = multiprocessing.cpu_count()
        else:
            self.n_workers = n_workers
        self.global_net = ACNet(mother=self,
                                scope=self.global_scope,
                                sess=self.sess,
                                globalmodel=None,
                                con_a_opt=None,
                                dis_a_opt=None,
                                c_opt=None,
                                training=learning)

        self.global_running_r = list()
        self.global_running_round_r = list()
        self.global_ep = 0

        self.web_shift = 0

        self.opt_a_dis = tf.train.AdamOptimizer(self.lr_a_dis, name='AdamAdis')
        self.opt_a_con = tf.train.AdamOptimizer(self.lr_a_con, name='AdamAcon')
        self.opt_c = tf.train.AdamOptimizer(self.lr_c, name='AdamC')

        if hiring:
            self.hiring()

        self.load_sess()
        self.global_net.sess = self.sess

        if self.workers is not None:
            for worker in self.workers:
                worker.AC.global_model = self.global_net
                worker.AC.sess = self.sess
                worker.AC.pull_global()
                print(worker.name, 'pulled')
        self.coord = tf.train.Coordinator()
        self.output_graph = True




    def dump_tensorboard(self):
        if self.output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir, ignore_errors=True)
            tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def load_sess(self):
        global_net_variables = [v for v in self.global_net.params]
        self.saver = tf.train.Saver(global_net_variables, name='saver', max_to_keep=15, allow_empty=True)
        self.sess = tf.Session()
        try:
            print('self.ckpt_path', self.ckpt_path)
            self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            # if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('self.global_net.global_step', self.sess.run(self.global_net.global_step))
            # print('self.global_net.global_step', self.sess.run(self.global_net.global_step))
            # global_step = self.global_net.global_step.eval(session=self.sess)
            logging.error('ckpt read')

        except (tf.errors.NotFoundError, ValueError, AttributeError) as e:
            print(e)
            self.sess.run(tf.global_variables_initializer())
            logging.error('no ckpt read')

    def dump_sess(self, extra_path=''):
        path = self.ckpt_path + extra_path
        global_step = self.sess.run(tf.assign_add(self.global_net.global_step, 1))
        print('global_step:', global_step)

        try:
            if self.global_net.global_step is not None:
                self.saver.save(self.sess, path, global_step=global_step)
            else:
                self.saver.save(self.sess, path)
            logging.error('ckpt saved: {}'.format(path))
        except:
            time.sleep(5)
            if self.global_net.global_step is not None:
                self.saver.save(self.sess, path, global_step=self.sess.run(tf.assign_add(self.global_net.global_step, 1)))
            else:
                self.saver.save(self.sess, path)
            logging.error('ckpt saved: {}'.format(path))

    def hiring(self):

        with tf.device("/cpu:0"):
            self.workers = list()
            # Create worker
            for i in range(self.n_workers):
                i_name = 'W_%i' % i  # worker name
                self.workers.append(Worker(mother=self,
                                           name=i_name,
                                           sess=self.sess,
                                           globalmodel=self.global_net,
                                           con_a_opt=self.opt_a_con,
                                           dis_a_opt=self.opt_a_dis,
                                           c_opt=self.opt_c,
                                           learning=self.learning
                                           ))
                print('hired mr. {}.'.format(i_name))


    def train(self, opposite_agents, max_global_ep=100, update_iter=137, gamma=0.99, dump_global_iter=3571, web=True, oppositenum=9, uris=list(), names=list()):

        self.update_iter = update_iter
        self.dump_global_iter = dump_global_iter

        self.dump_tensorboard()
        worker_threads = list()
        for i, worker in enumerate(self.workers):
            if web:
                def job():
                    try:
                        worker.webwork(opposite_agents=opposite_agents.copy(),
                                       max_global_ep=max_global_ep,
                                       update_iter=update_iter,
                                       gamma=gamma,
                                       dump_global_iter=dump_global_iter,
                                       oppositenum=oppositenum,
                                       uri=uris[i],
                                       name=names[i])
                    except ValueError as e:
                        while True:
                            print(e)
                            # self.sess.run(worker.AC.pull_global())
                            worker.webwork(opposite_agents=opposite_agents.copy(),
                                           max_global_ep=max_global_ep,
                                           update_iter=update_iter,
                                           gamma=gamma,
                                           dump_global_iter=dump_global_iter,
                                           oppositenum=oppositenum,
                                           uri=uris[i],
                                           name=names[i])
            else:
                def job():
                    try:
                        worker.work(opposite_agents=opposite_agents.copy(),
                                    max_global_ep=max_global_ep,
                                    update_iter=update_iter,
                                    gamma=gamma,
                                    dump_global_iter=dump_global_iter,
                                    oppositenum=oppositenum)
                    except ValueError as e:
                        while True:
                            print(e)
                            # self.sess.run(worker.AC.pull_global())
                            worker.work(opposite_agents=opposite_agents.copy(),
                                        max_global_ep=max_global_ep,
                                        update_iter=update_iter,
                                        gamma=gamma,
                                        dump_global_iter=dump_global_iter,
                                        oppositenum=oppositenum)

            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        self.coord.join(worker_threads)

    def getReload(self, state):
        return self.global_net.getReload(state)

    def takeAction(self, state, playerid):
        return self.global_net.takeAction(state, playerid)

    def new_round(self, state, playerid):
        return self.global_net.new_round(state, playerid)

    def round_end(self, state, playerid):
        return self.global_net.round_end(state, playerid)

    def game_over(self, state, playerid):
        return self.global_net.game_over(state, playerid)

    def game_start(self, state, playerid):
        return self.global_net.game_start(state, playerid)

    def single_train(self, opposite_agents, max_global_ep=100, dump_global_iter=3000, update_iter=1, web=False,
                    oppositenum=5, uris=list(), names=list()):
        self.train(opposite_agents,
                   max_global_ep=max_global_ep,
                   dump_global_iter=dump_global_iter,
                   update_iter=update_iter,
                   web=web,
                   oppositenum=oppositenum,
                   uris=uris,
                   names=names)  # 3571
        self.dump_sess()
        self.global_ep = 0

if __name__ == '__main__':
    from agent import allinModel
    from agent import allFoldModel
    from agent import allRaiseModel
    from agent import allCallModel
    from agent import randomAgent
    import sys
    sys.path.append('../../')
    from agent.MonteCarlo.agent import NpRandom
    model_dir = 'C:/Users/shanger_lin/Desktop/models/A3CAgent/model100'
    # agent = A3CAgent(model_dir=model_dir, learning=True, hiring=True, n_workers=16, dump_global_iter=100, update_iter=10)
    # simple_o_list = ([allCallModel()] * 10) + ([allRaiseModel()] * 2) + ([allinModel()] * 2) #+ ([allFoldModel()] * 1) + ([allinModel()] * 2)
    #
    # o_list = [NpRandom(None, 'omggyy', timeout=0.25, cores=1),
    #           NpRandom(None, 'omggyy2', timeout=0.25, cores=1),
    #           NpRandom(None, 'omggyy3', timeout=0.25, cores=1),
    #           NpRandom(None, 'omggyy4', timeout=0.25, cores=1),
    #           NpRandom(None, 'omggyy5', timeout=0.25, cores=1),
    #           NpRandom(None, 'omggyy6', timeout=0.5, cores=1),
    #           NpRandom(None, 'omggyy7', timeout=0.5, cores=1),
    #           NpRandom(None, 'omggyy8', timeout=0.5, cores=1),
    #           NpRandom(None, 'omggyy9', timeout=0.5, cores=1),
    #           NpRandom(None, 'omggyyT', timeout=0.25, cores=1),
    #           ] + simple_o_list
    # agent.single_train(opposite_agents=simple_o_list,
    #                    max_global_ep=1500,
    #                    dump_global_iter=300,
    #                    update_iter=1,
    #                    web=False,
    #                    oppositenum=5)
    # agent.single_train(opposite_agents=simple_o_list,
    #                    max_global_ep=1500,
    #                    dump_global_iter=300,
    #                    update_iter=3,
    #                    web=False,
    #                    oppositenum=5)
    # agent.single_train(opposite_agents=simple_o_list,
    #                    max_global_ep=10000,
    #                    dump_global_iter=1000,
    #                    update_iter=10,
    #                    web=False,
    #                    oppositenum=5)
    # agent.single_train(opposite_agents=o_list,
    #                    max_global_ep=10000,
    #                    dump_global_iter=300,
    #                    update_iter=5,
    #                    web=False,
    #                    oppositenum=5)


    uris = ['ws://poker-battle.vtr.trendnet.org:3001'] + \
           ['ws://poker-training.vtr.trendnet.org:3001/'] * len(range(1, 19)) * 2

    names = ['1886368b064b4b76be10d54d38958ce3'] +\
            ['omgg{}'.format(str(i)) for i in range(1, 19)] + \
            ['omggy{}'.format(str(i)) for i in range(1, 19)]

    assert len(uris) == len(names)
    agent = A3CAgent(model_dir=model_dir, learning=True, hiring=True, n_workers=len(names), dump_global_iter=100,
                     update_iter=10)


    agent.single_train(opposite_agents=list(),
                       max_global_ep=20000000000,
                       dump_global_iter=int(2*len(names)),
                       update_iter=1,
                       web=True,
                       oppositenum=9,
                       uris=uris,
                       names=names, #
                       )

