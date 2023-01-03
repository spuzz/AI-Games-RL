import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations = 1):
    value = np.zeros(env.n_states, dtype=np.float)

    for s in range(value.shape[0]):
        value[s] = 0
    changeInValue = 0
    for iter in range(max_iterations):
        for s in range(value.shape[0]):
            v = value[s]
            for a in range(env.n_actions):
                if a == policy[s]:
                    sumNextState = 0
                    for sNext in range(env.n_states):
                        if sNext != s:
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
        value = policy_evaluation(env, policy, gamma, theta, 5)
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
            changeInValue = max(changeInValue,  abs((v - value[s])))
        if changeInValue < theta:
            break
    policy = np.zeros(env.n_states, dtype=int)

    policy = policy_improvement(env, policy, value, gamma)

    return policy, value
