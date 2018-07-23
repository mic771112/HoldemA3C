import gym
import holdem
import agent

def lets_play(env, n_seats, model_list):
    init_stack = [3000 for i in range(6)]
    while True:

        state = env.reset()
        # env.render(mode='human')
        cycle_terminal = False
        # (cur_state)
        if env.episode_end:
            break
        # print(cur_state)

        print('init_stack', init_stack)

        while not cycle_terminal:
            # play safe actions, check when no one else has raised, call when raised.
            # print(">>> Debug Information ")
            # print("state(t)")
            # for p in cur_state.player_states:
            #     print(p)
            # print(cur_state.community_state)

            actions = holdem.model_list_action(state, n_seats=n_seats, model_list=model_list)
            # print(actions)
            next_state, rews, cycle_terminal, info = env.step(actions)

            # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
            # print(cur_state[0])

            state = next_state

            # print("reward(t+1)")
            # print(rews)
            # print("<<< Debug Information ")
            # env.render(mode="human")
        # print("final state")
        # print(cur_state)

        # total_stack = sum([p.stack for p in env._seats])
        # if total_stack != 10000:
        #     return
        stack = [i.stack for i in state.player_states]

        r = [i - j for i, j in zip(stack, init_stack)]
        print('stack', stack, sum(stack))
        print('r', r, sum(r), len(r))

        init_stack = stack

    print("Episode End!!!")

# env = holdem.TexasHoldemEnv(2)
env = holdem.TexasHoldemEnv(6) # gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(2)

model_list = list()

# start with 4 players
env.add_player(0, stack=3000)  # add a player to seat 0 with 1000 "chips"
model_list.append(agent.allRaiseModel())

env.add_player(1, stack=3000)  # add another player to seat 1 with 2000 "chips"
model_list.append(agent.allRaiseModel())

env.add_player(2, stack=3000)  # add another player to seat 2 with 3000 "chips"
model_list.append(agent.allRaiseModel())
#
env.add_player(3, stack=3000)  # add another player to seat 3 with 1000 "chips"
model_list.append(agent.allRaiseModel())
#
env.add_player(4, stack=3000)  # add another player to seat 3 with 1000 "chips"
model_list.append(agent.allRaiseModel())
#
env.add_player(5, stack=3000)  # add another player to seat 3 with 1000 "chips"
model_list.append(agent.allRaiseModel())

# env.add_player(6, stack=3000) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allCallModel())
#
# env.add_player(7, stack=3000) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(8, stack=3000) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allFoldModel())
#
# env.add_player(9, stack=3000) # add another player to seat 3 with 1000 "chips"
# model_list.append(agent.allinModel())

# play out a hand
lets_play(env, env.n_seats, model_list)