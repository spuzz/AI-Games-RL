import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations=1, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    iterations = 0
    for i in range(max_iterations):
        changeInValue = 0

        # for each possible state
        for s in range(value.shape[0]):
            # take a copy of the current value for state s
            v = value[s]
            sumNextState = 0
            # For each action available
            for a in range(env.n_actions):
                # Deterministic policy
                if a == policy[s]:
                    # for each possible next state
                    for sNext in range(env.n_states):
                        # calculate prob and reward from transition with action a
                        Pass = env.p(sNext, s, a)
                        Rass = env.r(sNext, s, a)
                        sumNextState += Pass * (Rass + (gamma * value[sNext]))
            value[s] = sumNextState
            # determine biggest change in value
            changeInValue = max(changeInValue, abs(v - value[s]))
        iterations += 1
        # if the biggest value change < theta then end loop
        if changeInValue < theta:
            break

    return value, iterations


def policy_improvement(env, policy, value, gamma):
    stable = True
    # for each state in policy
    for pol in range(policy.shape[0]):
        Qsa = np.zeros(env.n_actions, dtype=float)
        # for each possible action
        for a in range(Qsa.shape[0]):
            # for each possible next state
            for sNext in range(env.n_states):
                # calculate prob and reward from transition with action a
                Pass = env.p(sNext, pol, a)
                Rass = env.r(sNext, pol, a)
                Qsa[a] += Pass * (Rass + (gamma * value[sNext]))
        # if policy does not change then policy is stable
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
    value = np.zeros(env.n_states, dtype=np.float)
    total_iterations = 0
    # max_iterations has been applied to both policy iterations as a whole and policy evaluation
    for i in range(max_iterations):
        # interleave between policy evaluation and policy improvement
        value, iterations = policy_evaluation(env, policy, gamma, theta, max_iterations, value)
        total_iterations += iterations
        policy, stable = policy_improvement(env, policy, value, gamma)
        # if policy is stable then we end the algorithm
        if stable:
            break
    print("Number of iterations: " + str(total_iterations))
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    for i in range(max_iterations):
        changeInValue = 0
        # for each possible state
        for s in range(value.shape[0]):
            # take a copy of the current value for state s
            v = value[s]
            Qsa = np.zeros(env.n_actions, dtype=float)
            # for each possible action
            for a in range(Qsa.shape[0]):
                # for each possible next state
                for sNext in range(env.n_states):
                    # calculate prob and reward from transition with action a
                    Pass = env.p(sNext, s, a)
                    Rass = env.r(sNext, s, a)
                    Qsa[a] += Pass * (Rass + (gamma * value[sNext]))
            value[s] = max(Qsa)
            changeInValue = max(changeInValue, abs((v - value[s])))
        # if the biggest value change < theta then end loop
        if changeInValue < theta:
            break

    print("Number of iterations: " + str(i))
    policy = np.zeros(env.n_states, dtype=int)

    policy, stable = policy_improvement(env, policy, value, gamma)

    return policy, value
