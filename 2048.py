# -*- coding: utf-8 -*-

import gym_2048
import gym


if __name__ == '__main__':
  env = gym.make('2048-v0')
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  print(state_size,action_size)
  env.seed(42)

  print(env.reset().ravel())
  env.render()

  done = False
  moves = 0
  while moves<1:
    action = env.np_random.choice(range(4), 1).item()
    next_state, reward, done, info = env.step(action)
    print(next_state.ravel())
    moves += 1

    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

  print('\nTotal Moves: {}'.format(moves))