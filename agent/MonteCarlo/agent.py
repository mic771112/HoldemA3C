#! /usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging
import hashlib
import time
import datetime
import collections
import numpy as np
from pprint import pprint

from websocket import create_connection
from holdem import ACTION, action_table
from . import brain
from . import templates
from . import table
from . import card


EVENT_SPACE = {'__game_start',
               '__game_prepare',
               '__new_peer',
               '__new_peer_2',
               '__left',
               '__left_2',
               '_join',
               '__show_action',  # 
               '__deal',
               '__start_reload',
               '__round_end',  #
               '__new_round',  #
               '__bet',  #
               '__action',  #
               '__game_over',  #
               '__pairing',
               } 
ROUNDNAME = {'Deal', 'Flop', 'Turn', 'River'}
ACTION_SPACE = {"bet", "call", "check", "raise", "fold", "allin"}


class NpRandom:

    logger = logging.getLogger(__name__)

    def __init__(self, server, player, timeout=1.0, cores=4):
        self.timeout = timeout
        self.server = server
        self.player = player
        self.id = hashlib.md5(self.player.encode('utf-8')).hexdigest()
        # self.table = table.Table(self.id)

        # self.add_print_logger()
        self.add_file_logger()
        self.logger.setLevel(logging.WARNING)

        self.log_name_format = 'data/{}_{}.jl'
        self.brainlog_name_format = 'brain_log/{}_{}_{}.jl'

        self.pot = 0
        self.round_chip_init = None

        self.log_collector = None
        self.get_log_collector()

        self.brain_logger = None
        self.brain = brain.MonteCarloBrain(cores=cores)
        self.get_brain_logger()

        if server is not None:
            self.ws = create_connection(self.server)

    def get_log_collector(self):
        if self.log_collector is not None:
            if not self.log_collector.closed:
                self.log_collector.close()
        path = self.log_name_format.format(self.player,
                                           ''.join([i for i in datetime.datetime.now().isoformat() if i.isalnum()]))
        self.log_collector = open(path, 'w')

    def get_brain_logger(self):
        if self.brain_logger is not None:
            if not self.brain_logger.closed:
                self.brain_logger.close()
        path = self.brainlog_name_format.format(self.brain.__class__.__name__,
                                                self.player,
                                                ''.join([i for i in datetime.datetime.now().isoformat()
                                                         if i.isalnum()]))
        self.brain_logger = open(path, 'w')

    def add_file_logger(self):
        formatter = logging.Formatter('%(message)s')
        hdlr = logging.FileHandler('data_.log')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def add_print_logger(self):
        formatter = logging.Formatter('%(message)s')
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    @staticmethod
    def state2data(state, playerid):
        data = dict()
        data['tableNumber'] = 1
        data['game'] = {'smallBlind': {'amount': state.community_state.smallblind},
                        'board': [card.DEUCES2CARD[i] for i in state.community_card if i != -1],
                        'players': [{'isSurvive': p.playing_hand,
                                    'folded': not p.playing_hand,
                                     'playerName': {'cards': [card.DEUCES2CARD[i]
                                             for i in p.hand if i != -1]}}
                                   if seat ==playerid else {'isSurvive': p.playing_hand, 'folded': not p.playing_hand, 'playerName': dict()}
                                    for seat, p in enumerate(state.player_states)]}
        data["self"] = {'cards': [card.DEUCES2CARD[i] for i in state.player_states[playerid].hand if i != -1],
                        'minBet': state.community_state.to_call,
                        'chips': state.player_states[playerid].stack,
                        }
        # print(data)
        return data
    def takeAction(self, state, playerid):
        data = self.state2data(state, playerid)
        # action, amount, actionlog = self.brain.infer(data=data, pot=state.player_states[playerid].stack,
        #                                              timeout=self.timeout)
        try:
            action, amount, actionlog = self.brain.infer(data=data, pot=state.player_states[playerid].stack, timeout=self.timeout)
        except (ValueError, KeyError) as e:
            print('=====>', e)
            return ACTION(action_table.FOLD, 0)  ## exception for tablle lookup errror

        if action == 'fold':
            return ACTION(action_table.FOLD, int(amount))
        elif action == 'check':
            return ACTION(action_table.CHECK, int(amount))
        elif action == 'call':
            return ACTION(action_table.CALL, int(amount))
        elif action == 'bet':
            return ACTION(action_table.RAISE, int(amount))


        return

    def new_round(self, state, playerid):
        pass

    def round_end(self, state, playerid):
        pass

    def game_over(self, state, playerid):
        pass

    def getReload(self, state):
        return False

    def takeaction(self, event, data):

        if event == "__bet":
            action, amount, actionlog = self.brain.infer(data=data, pot=self.pot)
            print('action, amount', action, amount)
            self.ws.send(templates.action_message(player=self.id, action=action, amount=amount))
            self.log_collector.write(actionlog + '\n')

        elif event == "__action":  # "bet", "call", "check", "raise", "fold", "allin"
            # vector = self.table.get_table_feature_vector()
            # print('vector.shape', vector.shape)
            # print('vector.shape', vector)
            action, amount, actionlog = self.brain.infer(data=data, pot=self.pot)
            self.ws.send(templates.action_message(player=self.id, action=action, amount=int(amount)))
            print('action, amount', action, amount)
            self.log_collector.write(actionlog + '\n')
            pprint(templates.action_message(player=self.id, action=action, amount=amount))
        elif event == '__show_action':
            # self.table.update_from_show_action(data)
            self.pot = data['table']['totalBet']
            if data['action']['playerName'] == self.id:
                pprint(data['action'])

        elif event == '__deal':
            pass

        elif event == '__new_peer':
            pass

        elif event == '__start_reload':
            pass

        elif event == '__new_round':
            for p in data['players']:
                if p['playerName'] == self.id:
                    self.round_chip_init = p['chips']

        elif event == '__round_end':
            self.pot = 0

            amount = 0
            human_count = collections.Counter()

            self_rank = None
            opposite_rank = list()

            if self.round_chip_init is not None:
                for p in data['players']:

                    if p['playerName'] == self.id:

                        if p['winMoney']:
                            amount = p['chips'] - self.round_chip_init
                        else:
                            amount = p['chips'] - self.round_chip_init
                        if 'hand' in p:
                            self_rank = p['hand']['rank']
                    else:
                        if 'hand' in p:
                            human_count.update([p['isHuman']])
                            opposite_rank.append(p['hand']['rank'])

            if self_rank is not None:
                should_win = self_rank >= np.max(opposite_rank) if opposite_rank else True
                message = json.dumps({'win': str(amount > 0),
                                      'should_win': str(should_win),
                                      'amount': str(amount),
                                      'opposites_is_human': str(human_count),
                                      })
                self.brain_logger.write(message + '\n')
                print(message)

        elif event == '__game_over':
            # todo: got error here
            print(json.dumps({'game_over': data}))
            self.brain_logger.write(json.dumps({'game_over': data}) + '\n')
            self.get_log_collector()
            self.get_brain_logger()

            self.ws.send(templates.join_message(self.player))

    def dolisten(self):
        try:
            self.ws.send(templates.join_message(self.player))

            while True:
                result = self.ws.recv()
                # print('result', str(result))
                if not result:
                    self.ws.close()
                    print('ws disconnected, sleep for 10 minutes.')
                    time.sleep(60)
                    self.ws = create_connection(self.server)
                    self.ws.send(templates.join_message(self.player))

                msg = json.loads(result)
                self.logger.info(msg)
                self.log_collector.write(result + '\n')
                event_name = msg["eventName"]
                data = msg["data"]

                if event_name not in EVENT_SPACE:
                    self.logger.error('ERROR: %s not in EVENT_SPACE' % event_name)
                self.takeaction(event_name, data)

        except Exception as e:
            print(e)
            self.logger.info(e)
            self.dolisten()


if __name__ == '__main__':
    # pip install websocket-client
    # player = 'omg4'
    # server = 'ws://poker-training.vtr.trendnet.org:3001/'

    # server = "ws://poker-dev.wrs.club:3001/"

    # player = '92f63b8a10594e9e8fcda16770a329a3'
    player = '1886368b064b4b76be10d54d38958ce3'  # updated @ 20180719
    server = 'ws://poker-battle.vtr.trendnet.org:3001'
    # id = hashlib.md5(player.encode('utf-8')).hexdigest()

    print('server:', server, 'player:', player, 'id:', id)
    nprandom = NpRandom(server=server, player=player)
    nprandom.dolisten()
