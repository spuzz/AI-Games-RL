import numpy as np
import contextlib

class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env
        
        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']] # a list

        self.state_image = {self.env.absorbing_state: 
                            np.stack([np.zeros(lake.shape)] + lake_image)}
        
        index=0
        for state in range(lake.size):
            # TODO: 
            self.state_image[state] = np.stack([np.zeros(lake.shape)] + lake_image)
            index = index + 1 if (state % lake.shape[0] == 0 and state != 0) else index
            # encodes the first channel to set the agent location to the given state
            self.state_image[state][0][index][state % lake.shape[0]] = 1

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None, title=""):
        self.env.render(policy, value, title)