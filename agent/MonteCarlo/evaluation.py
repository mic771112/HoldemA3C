from __future__ import division

import logging
from time import time

from card import cards2deuces, draw_cards
from deuces.evaluator import Evaluator
from multiprocessing import Process, Queue
from deuces import Card

CardEvaluator = Evaluator()

def single_infer(hand: list, board: list, opponents_count: int):

    shuffle_cards = [i for i in draw_cards(deuces=True) if i not in set(hand+board)]

    if 5 - len(board) > 0:
        for _ in range(5 - len(board)):
            board += [shuffle_cards.pop()]

    opponents_cards = [shuffle_cards[2*i: 2*i+2] for i in range(opponents_count)]
    return hand, board, opponents_cards


def check_if_win(hand, board, opponents_cards):
    score = CardEvaluator.evaluate(cards=hand, board=board)
    player_scores = tuple(map(lambda x: CardEvaluator.evaluate(cards=x, board=board), opponents_cards))

    if not opponents_cards:
        return True

    if score <= min(player_scores):
        return True
    else:
        return False


def get_win_possibility(hand, board, opponents_count, timeout=None, iterout=None):
    assert [timeout, iterout].count(None) == 1

    w = 0
    a = 0
    t0 = time() # time() is faster than datetime.now()
    if timeout is not None:

        while time() - t0 < timeout:
            win = check_if_win(*single_infer(hand, board, opponents_count=opponents_count))
            a += 1
            if win:
                w += 1

    elif iterout is not None:

        while a <= iterout:
            win = check_if_win(*single_infer(hand, board, opponents_count=opponents_count))
            a += 1
            if win:
                w += 1

            if a and not a % 3000:
                print(hand, board)
                print(a, w/a, time() - t0)
                a = w = 0
    return w, a


def card2winrate(hands, board, opponents_count, timeout=1.5):
    wins, count = get_win_possibility(hand=cards2deuces(hands), board=cards2deuces(board), opponents_count=opponents_count, timeout=timeout)
    return wins/count, count



class MultiProcessedProbability(object):

    def __init__(self, cores=4):
        self.inqueue = Queue()
        self.outqueue = Queue()

        self.ps = [Process(target=MultiProcessedProbability._process, args=(self.inqueue, self.outqueue)) for _ in range(cores)]
        for p in self.ps:
            p.start()
        logging.info("%s processes initialized", cores)

    @staticmethod
    def _process(inqueue, outqueue):
        while True:
            task = inqueue.get()
            if task:
                cards, board, opponents_count, timeout = task
                result = get_win_possibility(cards, board, opponents_count, timeout)
                outqueue.put(result)
            else:
                break

    def winrate(self, cards, board, opponents_count, timeout):

        cards = cards2deuces(cards)
        board = cards2deuces(board)

        for _ in self.ps:
            self.inqueue.put((cards, board, opponents_count, timeout))

        w, a = 0, 0
        for _ in self.ps:
            wins, count = self.outqueue.get()
            w += wins
            a += count
        return w/a, a

    def close(self):
        for _ in self.ps:
            self.inqueue.put(None)
        for p in self.ps:
            p.join()

HOLECARDSCORE = {v: i for i, v in enumerate('23456789TJQKA')}
def hole_cards_score(hole_cards):
    """
    Calculate a score for hole cards
    Return hish score if we got high cards/pair/possible straight/possible flush
    """
    high_card = 0
    same_suit = 0
    possible_straight = 0
    pair = 0

    base_score = HOLECARDSCORE[hole_cards[0][0]] + HOLECARDSCORE[hole_cards[1][0]]
    if base_score > 20:
        high_card = base_score - 20

    if hole_cards[0][1] == hole_cards[1][1]:
        same_suit = 2

    value_diff = HOLECARDSCORE[hole_cards[0][0]] - HOLECARDSCORE[hole_cards[1][0]]
    if value_diff in [-4, 4]:
        possible_straight = 1
    if value_diff in [-3, 3]:
        possible_straight = 2
    if value_diff in [-2, -1, 1, 2]:
        possible_straight = 3
    if value_diff == 0:
        pair = 10

    return (pair + same_suit + high_card + possible_straight) * base_score


def expected_value(pot, win_prob, min_bet):
    """
    Compute expacted value to attend next stage
    """
    ev = (((pot + min_bet) * win_prob) - min_bet)
    # print("==== Expected value ==== %d" % ev)
    return ((pot + min_bet) * win_prob) - min_bet

if __name__ =='__main__':

    # example: cards == "KH","5C","5D"

    # print(card2winrate(hands=['5S', '5H'], board=["KH", "5C", "5D"], opponents_count=5, timeout=1.5))

    p = MultiProcessedProbability(4)
    print(p.winrate(cards=['5S', '5H'], board=["KH", "5C", "5D"], opponents_count=5, timeout=150))
    p.close()
    # print(get_win_possibility(hand, board, opponents_count=5, timeout=1.5))
