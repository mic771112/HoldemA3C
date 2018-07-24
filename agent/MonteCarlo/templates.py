JOIN_TEMPLATE = '{"eventName": "__join", "data": {"playerName": "%s"}}'
RELOAD_TEMPLATE = '{"eventName": "__reload"}'

# CALL_TEMPLATE = '{"eventName": "__action", "data": {"action": "call", "playerName": "%s"}}'
# BET_TEMPLATE = '{"eventName": "__action", "data": {"action": "bet", "playerName": "%s", "amount": %s}}'
# ACTION_TEMPLATE = '{"eventName": "__action", "data": {"action": "%s", "playerName": "%s", "amount": %s}}'

CALL_TEMPLATE = '{"eventName": "__action", "data": {"action": "call", "playerName": "%s"}}'
BET_TEMPLATE = '{"eventName": "__action", "data": {"action": "bet", "playerName": "%s", "amount": %s}}'
ACTION_TEMPLATE = '{"eventName": "__action", "data": {"action": "%s", "playerName": "%s", "amount": %s}}'


def action_message(player, action, amount=0):
    return ACTION_TEMPLATE % (action, player, amount)


def call_message(player):
    return CALL_TEMPLATE % player


def bet_message(player, amount):
    return BET_TEMPLATE % (player, int(amount))


def join_message(player):
    return JOIN_TEMPLATE % player


def _reload_message():
    return RELOAD_TEMPLATE