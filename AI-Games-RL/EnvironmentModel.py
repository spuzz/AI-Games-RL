import numpy as np


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward

    def render(self):
        raise NotImplementedError()

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1. / n_states)
        self.n_steps = 0
        self.totalReward = 0
        #self.state = self.random_state.choice(self.n_states, p=self.dist)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def step(self, action):
        if action < 0 or action > self.n_actions:
            raise Exception("Invalid action")

        self.n_steps += 1
        done = self.n_steps > self.max_steps

        self.state, reward = self.draw(self.state, action)
        self.totalReward += reward
        return self.state, reward, done
