from itertools import cycle
import numpy as np

ROUND_NAMES = {r: i for i, r in enumerate(('Deal', 'Flop', 'Turn', 'River'))}

class Table:

    def __init__(self, id):
        self.id = id
        self.profile = dict()
        self.bb = 20

        # self.pot = None
        # self.min_bet = None
        #
        # self.players = None  # 9
        # self.stacks = None  # 9
        # self.chips = None  # 9
        # self.valid = None

    def update_from_new_round(self, data):
        for p in data['players']:
            pass

    def update_from_round_end(self, data):
        pass

    def update_from_show_action(self, data):
        self.bb = data['table']['bigBlind']['amount']
        self.profile = dict()
        for p in data['players']:
            # if p['playerName'] == self.id:
            #     continue
            # if p['folded']:
            #     continue
            if not p['isSurvive']:
                continue

            stack = int(p['bet']) + int(p['roundBet'])
            self.profile[p['playerName']] = {'stack': int(stack),
                                             'pot': int(data['table']['totalBet']) - stack,
                                             'chips': int(p['chips']),
                                             'isHuman': int(p['isHuman']),
                                             'isOnline': int(p['isOnline'])}
    @staticmethod
    def chip_normalize(chips, maxchip=3000):
        return (chips - (maxchip/2)) / maxchip

    def get_table_feature_vector(self):
        print(1)
        opposite_vectors = list()
        if not self.profile:
            return np.zeros(42)
        print(2)
        for k, v in cycle(self.profile.items()):
            print(3)
            if k == self.id:
                continue
            else:
                pvector = (v['pot'] / (v['stack'] + 1),
                           self.chip_normalize(v['stack']),
                           self.chip_normalize(v['pot']),
                           self.chip_normalize(v['chips']),
                           # v['isHuman'],
                           # v['isOnline'],
                           )
                opposite_vectors.append(pvector)

            if len(opposite_vectors) >= 9:
                break
        print(4)
        if self.id in self.profile:
            svector = np.array((self.profile[self.id]['pot'] / (self.profile[self.id]['stack'] + 1),
                                self.chip_normalize(self.profile[self.id]['stack']),
                                self.chip_normalize(self.profile[self.id]['pot']),
                                self.chip_normalize(self.profile[self.id]['chips']),
                                self.profile[self.id]['isHuman'],
                                self.profile[self.id]['isOnline'],
                                ))
        else:
            svector = None
        ovector = np.concatenate(opposite_vectors)
        print(5)
        return np.concatenate([svector, ovector])



