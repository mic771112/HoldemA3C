import holdem
import agent

# SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001" # beta
# SERVER_URI = r"ws://allhands2018-training.dev.spn.a1q7.net:3001" # training
SERVER_URI = 'ws://poker-training.vtr.trendnet.org:3001/'
# name="Enter Your Name Here"
name = "omg"
model = agent.allCallModel()

# while True: # Reconnect after Gameover
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=False, playing_live=False)
client_player.doListen()