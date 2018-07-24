import abc
import json
import numpy as np

from . import evaluation
from .evaluation import MultiProcessedProbability, hole_cards_score, expected_value


class Brain(abc.ABC):
    def __init__(self):
        self.loop_count = 0
        self.state_cache = 0
        self.hand_cache = 0

    @staticmethod
    def get_sb(data):
        return data['game']['smallBlind']['amount']

    @staticmethod
    def get_board(data):
        return data['game']['board']

    @staticmethod
    def get_hands(data):
        return data['self']['cards']

    @staticmethod
    def get_minbet(data):
        return data["self"]["minBet"]

    @ staticmethod
    def get_chip(data):
        return data["self"]['chips']

    @staticmethod
    def get_self(data):
        return data["self"]

    @staticmethod
    def get_opposite_count(data):
        return len([p for p in data['game']['players']
                    if (
                            p['isSurvive'] and (not p['folded']) and (not ('cards' in p['playerName']))
                    )])

    @staticmethod
    def get_expected_reward(win_possibility, pot, min_bet):
        return win_possibility * (pot + min_bet) - min_bet  # win_possibility*pot - (1-win_possibility) - min_bet

    @classmethod
    @abc.abstractmethod
    def infer(self, data, pot,):
        return 'check', 0


class MonteCarloBrain(Brain):

    def __init__(self, cores=4):
        super().__init__()
        self.simulation = MultiProcessedProbability(cores=cores)

    def infer(self, data, pot, timeout=1):
        opposite_count = self.get_opposite_count(data)
        hands = self.get_hands(data)
        board = self.get_board(data)
        min_bet = int(self.get_minbet(data))
        chips = int(self.get_chip(data))
        sb = int(self.get_sb(data))
        sb = min(sb, 160)
        # print(opposite_count)
        if self.hand_cache == hands and (self.state_cache == len(board)):
            self.loop_count += 1
        else:
            self.hand_cache = hands
            self.state_cache = len(board)
            self.loop_count = 0

        # win_possibility, iter_count = evaluation.card2winrate(hands=hands,
        #                                                       board=board,
        #                                                       opponents_count=opposite_count,
        #                                                       timeout=timeout)

        win_possibility, iter_count = self.simulation.winrate(cards=hands,
                                                              board=board,
                                                              opponents_count=opposite_count,
                                                              timeout=timeout)

        expected_reward = self.get_expected_reward(win_possibility, pot, min_bet)
        table = data['tableNumber']
        to_write = {'table': table,
                    'opposite_count': opposite_count,
                    'hands': hands,
                    'board': board,
                    'min_bet': min_bet,
                    'chips': chips,
                    'win_possibility': win_possibility,
                    'expected_reward': expected_reward,
                    'self.pot': pot,
                    'iter_count': iter_count,
                    }

        # print('to_write', to_write)
        actionlog = json.dumps({'action_check': to_write})

        board_len = len(board)
        if board_len == 0:

            ev = int(expected_value(pot=pot, win_prob=1/opposite_count, min_bet=min_bet))

            score = hole_cards_score(hands)
            if score >= hole_cards_score(["Ks", "Kh"]):
                return 'bet', 3000, actionlog

            if score > min(hole_cards_score(["As", "Js"]), hole_cards_score(["Qs", "Qh"])):
                amount = min(int(2 * (min_bet + 10)), ev)
                
                if (amount > min_bet) and (self.loop_count < 2):
                    # print('bet', amount, actionlog)
                    return 'bet', amount, actionlog
                elif min_bet <= 8*sb:
                    # print('call', 0, actionlog)
                    return 'call', 0, actionlog
                # print('fold', 0, actionlog)
                return 'fold', 0, actionlog
            elif score > min(hole_cards_score(["As", "9s"]), hole_cards_score(["7s", "7h"])):
                amount = min(int(1.5 * (min_bet + 10)), ev)
                if (amount > min_bet) and (self.loop_count < 2):
                    # print('bet', amount, actionlog)
                    return 'bet', amount, actionlog
                elif min_bet < 6*sb:
                    # print('call', 0, actionlog)
                    return 'call', 0, actionlog
                # print('fold', 0, actionlog)
                return 'fold', 0, actionlog

            elif (ev >= 0) and (min_bet <= sb*4):
                # print('call', 0, actionlog)
                return 'call', 0, actionlog

            elif min_bet <= sb*2.5:
                # print('call', 0, actionlog)
                return 'call', 0, actionlog

            if min_bet == 0:
                # print('check', 0, actionlog)
                return 'check', 0, actionlog
            # print('fold', 0, actionlog)
            return 'fold', 0, actionlog


            # win_possibility = (win_possibility * 0.7 + (1 / opposite_count) * 0.3)
            #
            # if win_possibility > 0.6:
            #
            #     print('bet', min(sb*4 + min_bet, expected_reward), actionlog)
            #     return 'bet', min(sb*4 + min_bet, expected_reward), actionlog
            #
            # elif win_possibility > 0.4:
            #     print('bet', min(sb + min_bet, expected_reward), actionlog)
            #     return 'bet', min(sb + min_bet, expected_reward), actionlog
            #
            # elif (win_possibility > 0.1) and (min_bet <= int(sb*4)):
            #     print('call', 0, actionlog)
            #     return 'call', 0, actionlog
            #
            # elif min_bet <= 20 or expected_reward > 0:
            #     print('call', 0, actionlog)
            #     return 'call', 0, actionlog
            # else:
            #     print('fold', 0, actionlog)
            #     return 'fold', 0, actionlog
        sb = min(sb, 80)
        if win_possibility > 0.5:

            # amount = win_possibility * chips * board_len/5
            if board_len == 5:
                length_weight = 1
            elif board_len == 4:
                length_weight = 0.2
            else:
                length_weight = 0.1

            if win_possibility >= 0.9:
                amount = int(min(expected_reward, (data["self"]["chips"] * length_weight + data["self"]["minBet"])))

            elif win_possibility >= 0.8:
                amount = int(min(expected_reward, (0.7 * data["self"]["chips"] * length_weight + data["self"]["minBet"])))

            elif win_possibility >= 0.6:
                amount = int(min(expected_reward, (0.1 * data["self"]["chips"] * length_weight + data["self"]["minBet"])))

            else:
                amount = int(min(expected_reward, (0.05 * data["self"]["chips"] * length_weight + data["self"]["minBet"])))

            if amount > min_bet:
                if self.loop_count < 2:
                    # print('bet', amount, actionlog)
                    return 'bet', amount, actionlog
                else:
                    # print('call', 0, actionlog)
                    return 'call', 0, actionlog
            else:
                if win_possibility >= 0.8:
                    # print('check', 0, actionlog)
                    return 'check', 0, actionlog
                # print('fold', 0, actionlog)
                return 'fold', 0, actionlog

        elif win_possibility >= 0.4:
            if min_bet <= 6 * sb:
                # print('call', 0, actionlog)
                return 'call', 0, actionlog
            else:
                # print('fold', 0, actionlog)
                return 'fold', 0, actionlog

        elif expected_reward >= 0 and win_possibility >= 0.15:
            if min_bet <= 2 * sb:
                # print('check', 0, actionlog)
                return 'check', 0, actionlog

        # print('fold', 0, actionlog)
        return 'fold', 0, actionlog
        # if expected_reward >= 0:
        #
        #     to_bet = int(max(((chip / np.e) * np.exp(win_possibility)), 0))
        #     print('to_bet:', to_bet, 'min_bet:', min_bet, 'chip:', chip)
        #
        #     # todo: sometimes loos some amount event with all check check
        #     if to_bet < min_bet:
        #         print('fold')
        #         return 'fold', 0, actionlog
        #
        #     if to_bet == min_bet:
        #         print('check', 0)
        #         return 'check', 0, actionlog
        #
        #     else:
        #         print('bet', to_bet)
        #         return 'bet', to_bet, actionlog
        #
        # else:
        #     if 0 < min_bet:
        #         print('fold', 0)
        #         return 'fold', 0, actionlog
        #
        #     else:
        #         print('check', 0)
        #         return 'check', 0, actionlog



if __name__ == '__main__':
    a = Brain()