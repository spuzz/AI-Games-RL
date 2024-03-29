import numpy as np

from DeepQLearning import deep_q_network_learning
from FrozenLake import FrozenLake
from FrozenLakeImageWrapper import FrozenLakeImageWrapper
from TabularModelBasedRL import policy_iteration
from TabularModelBasedRL import value_iteration
from TabularModelFreeRL import sarsa, q_learning
from NonTabularModelFreeRL import LinearWrapper, linear_q_learning, linear_sarsa


def main():
    seed = 0

    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9

    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value, '## Policy iteration')

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value, '## Value iteration')

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
                          epsilon=0.5, seed=seed)
    env.render(policy, value, '## Sarsa')

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                                 epsilon=0.5, seed=seed)
    env.render(policy, value, '## Q-learning')

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma,
                               epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value, '## Linear Sarsa')

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma,
                                     epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value, '## Linear Q-learning')

    print('')

    image_env = FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                  gamma=gamma, epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value, '## Deep Q-network learning')


if __name__ == '__main__':
    main()


