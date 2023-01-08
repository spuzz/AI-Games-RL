import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations=1, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)

    for i in range(max_iterations):
        changeInValue = 0
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
    stable = True
    for pol in range(policy.shape[0]):
        Qsa = np.zeros(env.n_actions, dtype=float)
        for a in range(Qsa.shape[0]):
            for sNext in range(env.n_states):
                Pass = env.p(sNext, pol, a)
                Rass = env.r(sNext, pol, a)
                Qsa[a] += Pass * (Rass + (gamma * value[sNext]))
        newpol = np.argmax(Qsa)
        if newpol != policy[pol]:
            stable = False
        policy[pol] = np.argmax(Qsa)

    return policy, stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    value = np.zeros(env.n_states, dtype=float)
    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, 1, value)
        policy, stable = policy_improvement(env, policy, value, gamma)
        if stable:
            break
    #print("Number of iterations: " + str(i))
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)

    for i in range(max_iterations):
        changeInValue = 0
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

    #print("Number of iterations: " + str(i))
    policy = np.zeros(env.n_states, dtype=int)

    policy, stable = policy_improvement(env, policy, value, gamma)

    return policy, value
