
import torch
import torch.nn as nn
import torchvision.models as md


class Resnet_1(nn.Module):
  def __init__(self, action_space, input_shape, features):
    super(Resnet_1, self).__init__()
    resnet = md.resnet18(pretrained=True)

    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2

    self.flatten = nn.Flatten()

    with torch.no_grad():
      n_flatten = self.flatten(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(torch.zeros(1, *input_shape)))))))).shape[1]

    self.actor = nn.Sequential(
        nn.Linear(n_flatten, features),
        nn.ReLU(),
        nn.Linear(features, action_space)
    )

    self.critic = nn.Sequential(
        nn.Linear(n_flatten, features),
        nn.ReLU(),
        nn.Linear(features, 1)
    )


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.flatten(x)
    actor = self.actor(x)
    critic = self.critic(x)

    return actor, critic
