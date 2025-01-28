# MineRL Diamond Obtaining Agent

## Overview

This project aims to train a machine learning agent to play Minecraft and obtain a diamond  using advanced reinforcement learning techniques.The goal is to develop an AI agent that can effectively navigate the expansive Minecraft environment, gathering necessary resources leading up to the acquisition of a diamond.  The project leverages the MineRL dataset, which consists of human gameplay demonstrations, to aid in training the agent.

## Algorithms and Methods

### 1. K-Means Clustering
K-Means clustering for discretizing the action space. This technique allows to simplify the extensive action space found in Minecraft by creating a manageable set of actions through clustering human gameplay data.

### 2. ResNet-18 Model
We employed a modified ResNet-18 architecture for behavioral cloning. The architecture consists of convolutional layers that preprocess input frames, enabling the model to learn and predict actions based on human gameplay examples.

### 3. Dueling DQN (Deep Q-Network)
The project integrates Dueling DQN for reinforcement learning. This architecture helps enhance the learning efficiency by separating the state value and advantage of actions, improving the agent's learning capabilities in complex environments.

### 4. Soft Q Imitation Learning
A customized form of imitation learning was employed to allow the agent to learn from human demonstrations. This approach utilized the Dueling DQN framework and innovated the experience sampling and reward assignment to make the learning process more effective.

### 5. Hyperparameter Optimization
We selected specific hyperparameters such as a learning rate of 0.0001 and used the Adam optimizer to refine the model's performance. Hyperparameters were adjusted based on intuitive experimentation, considering the compute time available for training.


## Dataset

The **MineRL** dataset, which contains over 60 million state-action pairs captured from human gameplay. This dataset serves as a foundational resource for training the models and improving learning outcomes.


