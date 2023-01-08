import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from random import sample

class DeepQNetwork(torch.nn.Module):

    def __init__(self, env, learning_rate, kernel_size, conv_out_channels,
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0],
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels,
                                        out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features,
                                            out_features=env.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        # TODO: 
        # input tensor is passed through a convolutional layer and then flattened, 
        # it's then passed through a fully connected layer and a linear layer mapping to the actions
        x = F.relu(self.conv_layer(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_layer(x))
        return self.output_layer(x)

    def train_step(self, transitions, gamma, tdqn):
        mse = nn.MSELoss(reduction='sum')

        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)
        next_q = next_q.to(torch.float32)
        target = torch.Tensor(rewards) + gamma * next_q

        # TODO: the loss is the mean squared error between `q` and `target`
        # calculates the loss using mean squared error, then backpropagates it through the network
        self.optimizer.zero_grad()
        loss = mse(q, target)
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        # TODO:
        # samples a random batch of transitions of size batch_size
        batch = sample(list(self.buffer), batch_size)
        return batch
