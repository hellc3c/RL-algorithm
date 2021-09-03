import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import dtype, float16, float32, nn
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

    def compute_policy_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make a optimizer for updating the weights of policy network
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_funtion.parameters(), lr=lr)

    loss = nn.MSELoss()

    def train_one_epoch():
        # making empty lists for logging
        batch_obs = []
        batch_acts = []
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        done = False
        ep_rews = []

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collecting experience by acting in the enviroment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            obs, reward, done, _ = env.step(act)

            # save act and reward
            batch_acts.append(act)
            ep_rews.append(reward)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rews))

                # reset env to start a new iteration
                obs, done, ep_rews = env.reset(), False, []
                finished_rendering_this_epoch = True
                if len(batch_obs) > batch_size:
                    break
            
        # update parameter for both policy and value
        policy_optimizer.zero_grad()
        batch_policy_loss = compute_policy_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_policy_loss.backward()
        policy_optimizer.step()
        value_optimizer.zero_grad()
        batch_value_loss = loss(get_value(torch.as_tensor(batch_obs, dtype=torch.float32)).squeeze(), torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_value_loss.backward()
        value_optimizer.step()
        return batch_policy_loss, batch_value_loss, batch_rets, batch_lens

    for i in range(epochs):
        batch_policy_loss, batch_value_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t policy loss: %.3f \t value loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (i, batch_policy_loss, batch_value_loss, np.mean(batch_rets), np.mean(batch_lens)))
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('reward to go policy gradient.')
    train(env_name=args.env_name, lr=args.lr, render=False)
