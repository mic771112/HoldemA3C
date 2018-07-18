import gym
import holdem
import agent

def lets_play(env, n_seats, model_list):
    while True:
        cur_state = env.reset()
        env.render(mode='human')
        cycle_terminal = False
        # (cur_state)
        if env.episode_end:
            break
        print(cur_state)


        while not cycle_terminal:
            # play safe actions, check when no one else has raised, call when raised.
            # print(">>> Debug Information ")
            # print("state(t)")
            # for p in cur_state.player_states:
            #     print(p)
            # print(cur_state.community_state)

            actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)
            cur_state, rews, cycle_terminal, info = env.step(actions)

            # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
            print(cur_state[0])

            # print("reward(t+1)")
            # print(rews)
            # print("<<< Debug Information ")
            # env.render(mode="human")
        # print("final state")
        # print(cur_state)

        # total_stack = sum([p.stack for p in env._seats])
        # if total_stack != 10000:
        #     return

    print("Episode End!!!")

env = holdem.TexasHoldemEnv(2)
# env = gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(2)

model_list = list()

# start with 4 players
env.add_player(0, stack=123) # add a player to seat 0 with 1000 "chips"
model_list.append(agent.allFoldModel())

env.add_player(1, stack=234) # add another player to seat 1 with 2000 "chips"
model_list.append(agent.allFoldModel())

# env.add_player(2, stack=345) # add another player to seat 2 with 3000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(3, stack=456) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(4, stack=567) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(5, stack=678) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())

# env.add_player(6, stack=789) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allCallModel())
#
# env.add_player(7, stack=890) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(8, stack=901) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(9, stack=1012) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allinModel())

# play out a hand
lets_play(env, env.n_seats, model_list)