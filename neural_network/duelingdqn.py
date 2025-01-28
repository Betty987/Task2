import torch
import torch.nn as nn

class DuelingDQN:
  def __init__(self, input_size, output_size, action_space, features):
    self.input_size = input_size
    self.output_size = output_size
    super().__init__()

    self.conv1 = nn.Conv2d(input_size, 32, 8, stride=4)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(32, 64, 4, stide=2)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

    self.actor = nn.Sequential(
        nn.Linear(1024, features),
        nn.ReLU(),
        nn.Linear(features, action_space)
    )

    self.critic = nn.Sequential(
        nn.Linear(1024, features),
        nn.ReLU(),
        nn.Linear(features, 1)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.relu(x)

    actor = self.actor(x)
    critic = self.critic(x)

    return actor, critic
