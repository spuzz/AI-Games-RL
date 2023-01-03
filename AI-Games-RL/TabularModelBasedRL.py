import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations=1):
    value = np.zeros(env.n_states, dtype=np.float)

    for s in range(value.shape[0]):
        value[s] = 0
    changeInValue = 0
    for iter in range(max_iterations):
        for s in range(value.shape[0]):
            v = value[s]
            sumNextState = 0
            for a in range(env.n_actions):
                if a == policy[s]:
                    for sNext in range(env.n_states):
                        Pass = env.p(sNext, s, a)
                        Rass = env.r(sNext, s, a)
                        sumNextState += Pass * (Rass + (gamma * value[sNext]))
            value[s] = sumNextState
            changeInValue = max(changeInValue, abs(v - value[s]))

        if changeInValue < theta:
            break

    return value


def policy_improvement(env, policy, value, gamma):
    for pol in range(policy.shape[0]):
        Qsa = np.zeros(env.n_actions, dtype=float)
        for a in range(Qsa.shape[0]):
            for sNext in range(env.n_states):
                Pass = env.p(sNext, pol, a)
                Rass = env.r(sNext, pol, a)
                Qsa[a] += Pass * (Rass + (gamma * value[sNext]))
        policy[pol] = np.argmax(Qsa)

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float)
    for iter in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, policy, value, gamma)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    value = np.zeros(env.n_states, dtype=float)

    for s in range(value.shape[0]):
        value[s] = 0
    changeInValue = 0

    for iter in range(max_iterations):
        for s in range(value.shape[0]):
            v = value[s]
            Qsa = np.zeros(env.n_actions, dtype=float)
            for a in range(Qsa.shape[0]):
                for sNext in range(env.n_states):
                    Pass = env.p(sNext, s, a)
                    Rass = env.r(sNext, s, a)
                    Qsa[a] += Pass * (Rass + (gamma * value[sNext]))
            value[s] = max(Qsa)
            changeInValue = max(changeInValue, abs((v - value[s])))
        if changeInValue < theta:
            break
    policy = np.zeros(env.n_states, dtype=int)

    policy = policy_improvement(env, policy, value, gamma)

    return policy, value


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        done = False
        a = e_greedy(random_state, eta[i], q, s, env.n_actions)
        while not done:
            sNext, r, done = env.step(a)
            aNext = e_greedy(random_state, eta[i], q, s, env.n_actions)
            q[s][a] = q[s][a] + epsilon[i] * (r + (gamma * q[sNext][aNext]) - q[s][a])
            a = aNext
            s = sNext
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def e_greedy(random_state, eta, q, state, n_actions):
    p = np.array((1 - eta, eta))
    random = random_state.choice(2)
    if random == 1 or max(q[state] == 0):
        a = random_state.choice(n_actions)
    else:
        a = max(q[state])
    return int(a)
