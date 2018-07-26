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

    def __init__(self, model_dir, learning=True, hiring=True):

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

        self.lr_a = 1e-5
        self.lr_c = 1e-5

        self.update_global_iter = 100000
        self.dump_global_iter = 3571
        self.final_dumped = False
        self.n_workers = 8  # multiprocessing.cpu_count()
        self.global_net = ACNet(mother=self,
                                scope=self.global_scope,
                                sess=self.sess,
                                globalmodel=None,
                                a_opt=None,
                                c_opt=None)


        self.load_sess()
        self.global_net.sess = self.sess

        self.opt_a = tf.train.AdamOptimizer(self.lr_a, name='RMSPA')
        self.opt_c = tf.train.AdamOptimizer(self.lr_c, name='RMSPC')

        self.coord = tf.train.Coordinator()
        self.output_graph = True

        self.global_running_r = list()
        self.global_ep = 0

        self.web_shift = 0

        if hiring:
            self.hiring()

    def dump_tensorboard(self):
        if self.output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir, ignore_errors=True)
            tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def load_sess(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(name='saver', max_to_keep=15)

        try:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            # if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            logging.error('ckpt read')

        except (tf.errors.NotFoundError, ValueError, AttributeError):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            logging.error('no ckpt read')

    def dump_sess(self, global_step=None, extra_path=''):
        self.sess.run(tf.assign_add(global_step, 1))
        path = self.ckpt_path + extra_path

        try:
            if global_step is not None:
                self.saver.save(self.sess, path, global_step=global_step)
            else:
                self.saver.save(self.sess, path)
            logging.error('ckpt saved: {}'.format(path))
        except:
            time.sleep(5)
            if global_step is not None:
                self.saver.save(self.sess, path, global_step=global_step)
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
                                           a_opt=self.opt_a,
                                           c_opt=self.opt_c,
                                           learning=self.learning
                                           ))
            self.sess.run(tf.global_variables_initializer())
            self.load_sess()
            for worker in self.workers:
                worker.AC.pull_global()

    def train(self, opposite_agents, max_global_ep=100, update_global_iter=137, gamma=0.99, dump_global_iter=3571, web=True, oppositenum=9):

        self.update_global_iter = update_global_iter
        self.dump_global_iter = dump_global_iter

        self.dump_tensorboard()
        worker_threads = list()
        for worker in self.workers:
            if web:
                def job():
                    try:
                        worker.webwork(opposite_agents=opposite_agents.copy(),
                                       max_global_ep=max_global_ep,
                                       update_global_iter=update_global_iter,
                                       gamma=gamma,
                                       dump_global_iter=dump_global_iter,
                                       oppositenum=oppositenum)
                    except ValueError as e:
                        while True:
                            print(e)
                            # self.sess.run(worker.AC.pull_global())
                            worker.webwork(opposite_agents=opposite_agents.copy(),
                                           max_global_ep=max_global_ep,
                                           update_global_iter=update_global_iter,
                                           gamma=gamma,
                                           dump_global_iter=dump_global_iter,
                                           oppositenum=oppositenum)
            else:
                def job():
                    try:
                        worker.work(opposite_agents=opposite_agents.copy(),
                                    max_global_ep=max_global_ep,
                                    update_global_iter=update_global_iter,
                                    gamma=gamma,
                                    dump_global_iter=dump_global_iter,
                                    oppositenum=oppositenum)
                    except ValueError as e:
                        while True:
                            print(e)
                            # self.sess.run(worker.AC.pull_global())
                            worker.work(opposite_agents=opposite_agents.copy(),
                                        max_global_ep=max_global_ep,
                                        update_global_iter=update_global_iter,
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

    def single_train(self, opposite_agents, max_global_ep=100, dump_global_iter=3000, update_global_iter=1, web=False,
                    oppositenum=5):
        self.train(opposite_agents,
                   max_global_ep=max_global_ep,
                   dump_global_iter=dump_global_iter,
                   update_global_iter=update_global_iter,
                   web=web,
                   oppositenum=oppositenum)  # 3571
        self.dump_sess(global_step=agent.global_net.global_step)
        self.global_ep = 0

if __name__ == '__main__':
    from agent import allinModel
    from agent import allFoldModel
    from agent import allRaiseModel
    from agent import allCallModel
    from agent import randomAgent
    import sys

    agent = A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/model126', learning=True, hiring=True)
    # agent2 = A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/model125', learning=True, hiring=True)
    # agent = A3CAgent(model_dir='./model')
    # get_agent = lambda: A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/model125', learning=False, hiring=False)
    sys.path.append('../../')
    from agent.MonteCarlo.agent import NpRandom
    simple_o_list = [allCallModel()] * 10 \
                    + [allFoldModel()] * 1 \
                    + [allRaiseModel()] * 1 \
                    + [allinModel()] * 1

    o_list = [NpRandom(None, 'omggyy', timeout=0.25, cores=1),
              NpRandom(None, 'omggyy2', timeout=0.25, cores=1),
              NpRandom(None, 'omggyy3', timeout=0.25, cores=1),
              NpRandom(None, 'omggyy4', timeout=0.25, cores=1),
              NpRandom(None, 'omggyy5', timeout=0.25, cores=1),
              NpRandom(None, 'omggyy6', timeout=0.5, cores=1),
              NpRandom(None, 'omggyy7', timeout=0.5, cores=1),
              NpRandom(None, 'omggyy8', timeout=0.5, cores=1),
              NpRandom(None, 'omggyy9', timeout=1, cores=1),
              NpRandom(None, 'omggyyT', timeout=1, cores=1)]

    # o_list = [get_agent(), get_agent(), get_agent(), get_agent(), get_agent(), get_agent(), ]
    # o_list = [allCallModel(), allCallModel(), allCallModel(), allCallModel(), allCallModel(), allCallModel(), ]
    # o_list = [A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/modoel14', learning=True),
    #           A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/modoel14', learning=True)]


    # agent.single_train(opposite_agents=simple_o_list,
    #                    max_global_ep=10000,
    #                    dump_global_iter=1000,
    #                    update_global_iter=5,
    #                    web=False,
    #                    oppositenum=5)

    # agent.single_train(opposite_agents=o_list,
    #                    max_global_ep=500,
    #                    dump_global_iter=100,
    #                    update_global_iter=1,
    #                    web=False,
    #                    oppositenum=9)

    # agent.single_train(opposite_agents=o_list,
    #                    max_global_ep=1000,
    #                    dump_global_iter=100,
    #                    update_global_iter=3,
    #                    web=False,
    #                    oppositenum=9)

    agent.single_train(opposite_agents=o_list,
                       max_global_ep=10000,
                       dump_global_iter=100,
                       update_global_iter=5,
                       web=True,
                       oppositenum=9)
