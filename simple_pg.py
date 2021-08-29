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


# define training process
def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):  # the batch size number is big, why
    # make an enviroment via gym api
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This alogrithm only works for continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This alogrithm only works for discrete action spaces."

    obs_dim = env.observation_space.shape[0]  # guess it's not that general
    n_acts = env.action_space.n

    # construct a policy network
    # observation or state as input and action as output
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    print(logits_net)

    # calculate the action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        # print(f"logits: {logits}")
        return Categorical(logits=logits)

    # get action by sampler from the distribution get by the above funtion
    def get_action(obs):
        return get_policy(obs).sample().item()

    # compute the derivate term in the expected return formula
    # guess the parameter should be a series of each step
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make a optimizer for updating the weights of policy network
    optimizer = optim.Adam(logits_net.parameters(), lr=lr)

    # define the training process during the ineraction period
    def train_one_epoch():
        # making empty lists for logging
        batch_obs = []
        batch_acts = []
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []
        batch_lens = []

        # reset some variables at start of each episode
        obs = env.reset()  # sample a initial state from starting distribution
        done = False
        ep_rews = []  # list for rewards accrued throughout episode

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collecting experience by acting in the enviroment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()
            
            # save obs
            batch_obs.append(obs.copy())

            # act in enviroment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(act)

            # save action and reward
            batch_acts.append(act)
            ep_rews.append(reward)

            # if the game is over
            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                # print(f"total reward: {ep_ret}, step count: {ep_len}")
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset some varibale to continue training process
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop until we get enough data for a batch
                if len(batch_obs) > batch_size:
                    break
        
        # policy gradient update
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                    act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                    )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('simplest policy gradient.')
    train(env_name=args.env_name, lr=args.lr, render=True)
