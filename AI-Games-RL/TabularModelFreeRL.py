import numpy as np
from PlotReturns import PlotReturns

from PlotReturns import PlotReturns


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))
    eps = 0
    returnSum = np.zeros(max_episodes)
    for i in range(max_episodes):

        s = env.reset()
        done = False
        a = e_greedy(random_state, epsilon[i], q, s, env.n_actions)
        q_prev = q.copy()
        while not done:
            sNext, r, done = env.step(a)
            aNext = e_greedy(random_state, epsilon[i], q, sNext, env.n_actions)
            q[s][a] = q[s][a] + (eta[i] * (r + (gamma * q[sNext][aNext]) - q[s][a]))
            a = aNext
            s = sNext
            returnSum[i] += r + (gamma * q[sNext][aNext])
        if (q.argmax(axis=1) != q_prev.argmax(axis=1)).any():
            eps = i
    #PlotReturns(returnSum, "Sarsa Control")
    print("Episodes needed: " + str(eps))
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


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    dis_r_arr = [] #discounted returns array
    for i in range(max_episodes):
        done = False
        s = env.reset()
        dis_r = 0
        while not done:
            a = e_greedy(random_state, epsilon[i], q, s, env.n_actions)
            sNext, r, done = env.step(a)
            aNext = e_greedy(random_state, 0, q, sNext, env.n_actions)
            q[s][a] += eta[i] * (r + gamma * q[sNext][aNext] - q[s][a])
            s = sNext
            dis_r += r + gamma * q[sNext][aNext]
        dis_r_arr.append(dis_r)
    #PlotReturns(dis_r_arr, "Q Learning Control")
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value
