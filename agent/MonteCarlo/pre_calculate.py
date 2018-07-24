import pickle
from itertools import product, combinations

from evaluation import MultiProcessedProbability

def precalc(timeout, min_op=1, max_op=10, cores=4):
	p = MultiProcessedProbability(cores)

	all_cards = tuple(a+b for a, b in product('A23456789TJQK', "SHDC"))
	all_card_combinations = tuple(combinations(all_cards, 2))

	for cards in all_card_combinations:
		for opponents_count in range(min_op, max_op+1):
			card_a, card_b = cards
			prob, count = p.winrate(cards=cards, board=[], opponents_count=opponents_count, timeout=timeout)
			key = (card_a, card_b, opponents_count)
			value = (prob, count)
			print(key, value)
			yield key, value

	p.close()

if __name__ == "__main__":

	d = dict(precalc(2))

	with open("preflop-probabilities.p", "wb") as fw:
		pickle.dump(d, fw)
