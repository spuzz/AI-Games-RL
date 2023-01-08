import numpy as np
from PlotReturns import PlotReturns

from PlotReturns import PlotReturns


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None, title=""):
        self.env.render(policy, value, title)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    returnSum = np.zeros(max_episodes)
    for i in range(max_episodes):
        features = env.reset()

        q = features.dot(theta)
        done = False
        # Select action according to e-greedy policy
        a = e_greedy(random_state, epsilon[i], q, env.n_actions)
        while not done:
            sNext, r, done = env.step(a)
            # Calculate Q for next state
            qNext = sNext.dot(theta)

            # Select next action using qNext according to e-greedy policy
            aNext = e_greedy(random_state, epsilon[i], qNext, env.n_actions)

            # calculate temporal change
            sigma = (r + (gamma * qNext[aNext]) - q[a])

            # update weights using learning rate, temporal change and features for selected action
            theta += (eta[i] * sigma * features[a])
            a = aNext
            features = sNext
            q = qNext

            # Update array of return values
            returnSum[i] += r + (gamma * q[aNext])
    #PlotReturns(returnSum, "Linear Sarsa Control")
    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    dis_r_arr = [] #discounted returns array
    for i in range(max_episodes):
        features = env.reset()  # features = state representation
        done = False
        q = features.dot(theta)
        dis_r = 0
        while not done:
            action = e_greedy(random_state, epsilon[i], q, env.n_actions)
            features_prime, r, done = env.step(action)  # "features_prime" = next state
            delta = r - q[action]
            q_prime = features_prime.dot(theta)
            action_prime = e_greedy(random_state, 0, q_prime, env.n_actions)
            delta += gamma * q_prime[action_prime]
            theta += eta[i] * delta * features[action]
            features = features_prime
            q = q_prime
            dis_r += r + gamma * q_prime[action_prime]
        dis_r_arr.append(dis_r)
    #PlotReturns(dis_r_arr,"Linear Q Learning")
    return theta


def e_greedy(random_state, epsilon, q, n_actions):
    # Probability of exploitation vs exploration based on scaling epsilon
    p = np.array((1 - epsilon, epsilon))
    
    random = random_state.choice(2, p=p)
    if random == 1:
        a = random_state.choice(n_actions)
    else:
        # if more than one action has the same max (e.g initial values are all zero)
        # pick action at random from subset of max values
        max_index = np.argwhere(q == np.amax(q)).flatten()
        a = np.random.choice(max_index)
    return a
