import numpy as np

from FrozenLake import FrozenLake
from GridWorld import GridWorld
from TabularModelBasedRL import policy_iteration
from TabularModelBasedRL import value_iteration

if __name__ == '__main__':
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
    world = FrozenLake(lake, 0.1, 16)

    #grid = np.array([[0, 0, 0, 3], [0, 1, 0, 2], [0, 0, 0, 0]])
    #world = GridWorld(grid, 10, 10)
    actions = ['w','a','s','d']
    state = world.reset()
    world.step(2)

    done = False
    #policy, value = policy_iteration(world, 0.9, 1, 10, policy=None)
    policy, value = value_iteration(world, 0.9, 0.0001, 128, value=None)
    world.render(policy, value)
    while not done:
        c = actions[policy[world.state]]
        if c not in actions:
            print("Invalid Action")
            continue
        state, r, done = world.step(actions.index(c))
        world.render()


