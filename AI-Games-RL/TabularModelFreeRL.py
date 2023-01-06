import numpy as np


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    for i in range(max_episodes):

        s = env.reset()
        done = False
        a = e_greedy(random_state, epsilon[i], q, s, env.n_actions)
        while not done:
            sNext, r, done = env.step(a)
            aNext = e_greedy(random_state, epsilon[i], q, sNext, env.n_actions)
            q[s][a] = q[s][a] + (eta[i] * (r + (gamma * q[sNext][aNext]) - q[s][a]))
            a = aNext
            s = sNext
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def e_greedy(random_state, epsilon, q, state, n_actions):
    p = np.array((1 - epsilon, epsilon))
    random = random_state.choice(2, p=p)
    if random == 1:
        a = random_state.choice(n_actions)
    else:
        max_index = np.argwhere(q[state] == np.amax(q[state])).flatten()
        a = np.random.choice(max_index)
    return a


def q_learning(env, max_episodes , eta , gamma, epsilon , seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    for i in range(max_episodes):
        done = False
        s = env.reset()
        while not done:
            a = e_greedy(random_state, epsilon[i], q, s, env.n_actions)
            sNext, r, done = env.step(a)
            aNext = e_greedy(random_state, 0, q, sNext, env.n_actions)
            q[s][a] += eta[i]*(r+gamma*q[sNext][aNext]-q[s][a])
            s = sNext
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value



