import holdem
import agent
import time
import numpy as np

# SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001" # beta
# SERVER_URI = r"ws://allhands2018-training.dev.spn.a1q7.net:3001" # training
# server = 'ws://poker-dev.wrs.club:3001/'
server = 'ws://poker-training.vtr.trendnet.org:3001/'
player = "mamamia"

# model = agent.allinModel()
model = agent.A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/model51', learning=False, hiring=False)

while True: # Reconnect after Gameover
    # model.init
    client_player = holdem.ClientPlayer(server, player, model, debug=True, playing_live=True)
    client_player.doListen()
    time.sleep(np.random.randint(1, 20))