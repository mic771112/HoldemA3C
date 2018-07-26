from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table
import random

class randomModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''
        a = random.choice('rfc')
        if a == 'r':
            return ACTION(action_table.RAISE, state.community_state.to_call * 2)
        elif a == 'f':
            if state.community_state.to_call == 0:
                return ACTION(action_table.CHECK, 0)
            else:
                return ACTION(action_table.FOLD, 0)
        elif a == 'c':
            return ACTION(action_table.CHECK, 0)

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False