import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import float16, nn
from torch import optim
from torch.distributions.categorical import Categorical
from torch.random import seed

from utils import seed_torch

seed_torch(234)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation  # get the activation function by layer index
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


# compute reward to go expected return estimate value
# accumulate reward value from the current step
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    # iteration by reverse order
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This alogrithm only works for continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This alogrithm only works for discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    act_n = env.action_space.n

    policy_net = mlp(sizes=[obs_dim] + hidden_sizes + [act_n], activation=nn.Tanh, output_activation=nn.Identity)

    # use a simple mlp as a value funtion to estimate reward to go
    value_funtion = mlp(sizes=[obs_dim] + hidden_sizes + [1], activation=nn.Tanh, output_activation=nn.Identity)

    def get_policy(obs):
        logits = policy_net(obs)
        return Categorical(logits=logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()

    def get_value(obs):
        return value_funtion(obs)

    # compute the loss of value function's decent gradient
    def compute_value_loss(obs, rtgs):
        return (get_value(obs) - rtgs).mean()

    def compute_policy_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make a optimizer for updating the weights of policy network
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_funtion.parameters(), lr=lr)

    def train_one_epoch():
        
