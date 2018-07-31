import holdem
import agent

# SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001" # beta
# SERVER_URI = r"ws://allhands2018-training.dev.spn.a1q7.net:3001" # training
SERVER_URI = 'ws://poker-dev.wrs.club:3001/'
# SERVER_URI = 'ws://poker-training.vtr.trendnet.org:3001/'
# name="Enter Your Name Here"
name = "omgg"
# model = agent.allinModel()
model = agent.A3CAgent(model_dir='C:/Users/shanger_lin/Desktop/models/A3CAgent/model2', learning=True, hiring=False)

while True: # Reconnect after Gameover
    # model.init
    client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True, playing_live=True)
    client_player.doListen()