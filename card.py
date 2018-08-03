import numpy as np
from deuces import Card

#example: cards == "KH","5C","5D"

ROW_MAP = {v: i for i, v in enumerate('A23456789TJQK')}
COL_MAP = {v: i for i, v in enumerate('SHDC')}
ALL_CARDS = {i + 13*j: v+w for i, v in enumerate('A23456789TJQK') for j, w in enumerate('SHDC')}
CARD2INDEX = {c: i for i, c in ALL_CARDS.items()}

ALL_DEUCES = {i + 13*j: Card.new(v+w) for i, v in enumerate('A23456789TJQK') for j, w in enumerate('shdc')}
CARD2DEUCES = {c: ALL_DEUCES.get(i) for i, c in ALL_CARDS.items()}
DEUCES2CARD = {v: k for k, v in CARD2DEUCES.items()}
DEUCES2INDEX = {c: i for i, c in ALL_DEUCES.items()}

def cards2deuces(cards):
    return list(map(CARD2DEUCES.get, cards))


def deuces2cards(cards):
    return list(map(DEUCES2CARD.get, cards))


def draw_cards(deuces=True):
    if deuces:
        return map(ALL_DEUCES.get, np.random.permutation(52))
    else:
        return map(ALL_CARDS.get, np.random.permutation(52))


def cards2array(cards):
    card_array = np.zeros(shape=(14, 4))
    if not cards:
        return card_array
    rows, cols = zip(*list(map(lambda c: (ROW_MAP[c[0]], COL_MAP[c[1]]), cards)))
    card_array[rows, cols] = 1
    card_array[-1, :] = card_array[0, :]  # for tail A
    return card_array


def array_average_inferring(card_array, up_to=7):
    card_count = np.sum(card_array[:-1, :] == 1)
    dont_have_count = 52 - card_count
    card_array[np.where(card_array == 0)] = (up_to - card_count) / dont_have_count
    # note: np.sum(infer_possbility_for_array(get_card_array(cards))[:-1,:]) == up_to
    return card_array


def pooling(array, kernel, ptype='avg'):
    assert ptype in {'avg', 'max', 'sum'}
    # print(array.shape)
    # print(array)
    rows, cols = array.shape
    krows, kcols = kernel
    pooled_rows, pooled_cols = rows - krows + 1, cols - kcols + 1
    pooled_array = np.zeros(shape=(pooled_rows, pooled_cols))
    for i in range(pooled_rows):
        for j in range(pooled_cols):

            slice = array[i: i+krows, j: j+kcols]

            if ptype == 'avg':
                pooled_array[i, j] = np.mean(slice)
            elif ptype == 'max':
                pooled_array[i, j] = np.max(slice)
            elif ptype == 'sum':
                pooled_array[i, j] = np.sum(slice)

    return pooled_array


def array2features(array):
    two_three_four = pooling(array, kernel=(1, 4))  # 14 x 1
    flush = pooling(pooling(array, kernel=(13, 1)), kernel=(2, 1), ptype='max')   # 1 x 4
    straight_flush = pooling(pooling(array, kernel=(5, 1)), kernel=(1, 4), ptype='max')  # 10 x 1
    high = pooling(array, kernel=(1, 4), ptype='max')  # 14 x 1
    straight = pooling(pooling(array, kernel=(1, 4), ptype='max'), kernel=(5, 1))  # 10 x 1

    # two_three_four = two_three_four + 0.1 * np.max(two_three_four)
    # flush = flush + 0.3 * np.max(flush)
    # straight_flush = straight_flush + 0.5 * np.max(straight_flush)
    # high = high# + 0.1 * np.max(high)
    # straight = straight + 0.4 * np.max(straight)

    features = np.concatenate(tuple(map(lambda x: x.reshape(-1), (two_three_four,
                                                                  flush,
                                                                  straight_flush,
                                                                  high,
                                                                  straight))))
    return features  # 52


def cards2features(cards):
    return array2features(array_average_inferring(cards2array(cards)))


def deuces2features(deuces):
    return array2features(array_average_inferring(cards2array(deuces2cards(deuces))))


def deuces2array(deuces):
    return cards2array(deuces2cards(deuces))

def deuces2onehot(deuces):
    onehot = [0] * 52
    for d in deuces:
        onehot[DEUCES2INDEX[d]] = 1
    return onehot

if __name__ == "__main__":
    assert sorted(ALL_DEUCES.keys()) == list(range(52))
