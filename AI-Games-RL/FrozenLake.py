import contextlib
import numpy as np
import EnvironmentModel

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class FrozenLake(EnvironmentModel.Environment):
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    actionChance = 0

    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # Precalculate chance of taking selected action
        self.actionChance = 1 - self.slip

        self.state = np.where(self.lake_flat == '&')[0]
        EnvironmentModel.Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def step(self, action):
        state, reward, done = EnvironmentModel.Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):

        prob = 0

        # if in absorbing state then only absorbing state can be next
        if state == self.absorbing_state:
            if next_state == self.absorbing_state:
                return 1
            else:
                return 0

        # calculate grid location from 1d state
        x = int(state / self.lake.shape[1])
        y = int(state % self.lake.shape[1])

        # Check currently in hole/goal and if so only absorbing state can be next
        if self.lake[x][y] == '$' or self.lake[x][y] == '#':
            if next_state == self.absorbing_state:
                return 1
            else:
                return 0

        # Calculate probably of next_state given current state and action taken is the selected action
        prob += self.CheckStateProb(x, y, self.directions[action], state, next_state, self.actionChance)

        # Calculate probability of next_state for each possible direction taken via slip
        for a in range(self.n_actions):
            prob += self.CheckStateProb(x, y, self.directions[a], state, next_state, self.slip / 4)

        return prob

    def r(self, next_state, state, action):
        # if current state is absorbing reward is always 0
        if state == self.absorbing_state:
            return 0

        # calculate grid location from 1d state
        x = int(state / self.lake.shape[1])
        y = int(state % self.lake.shape[1])

        # Only change in reward is at the goal state
        if self.lake[x][y] == '$':
            return 1
        return 0

    def CheckStateProb(self, x, y, direction, state, next_state, chance):
        # direction taken
        x += direction[0]
        y += direction[1]
        prob = 0

        # if direction from action or slip is outside grid then only current state will gain probability
        if x < 0 or x >= self.lake.shape[0]:
            if next_state == state:
                prob += chance
            return prob

        if y < 0 or y >= self.lake.shape[1]:
            if next_state == state:
                prob += chance
            return prob

        # if next_state is reached for this direction then add probability
        if next_state == self.lake.shape[1] * x + y:
            prob += chance
        return prob

    def render(self, policy=None, value=None, title=""):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            lake = lake.reshape(self.lake.shape)

            print('Lake:')
            print(lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
            
            # print to output.txt
            with open("output.txt",'a') as textfile:

                print(title, file=textfile)
                print('', file=textfile)
                print('Lake:', file=textfile)
                print(lake, file=textfile)

                print('Policy:', file=textfile)
                print(policy.reshape(self.lake.shape), file=textfile)

                print('Value:', file=textfile)
                with _printoptions(precision=3, suppress=True):
                    print(value[:-1].reshape(self.lake.shape), file=textfile)
                print('', file=textfile)
    

