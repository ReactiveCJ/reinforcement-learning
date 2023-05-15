import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample).sum(dim=-1)
        action = torch.tanh(normal_sample)
        #log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(dim=1)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-1).sum(dim=-1)
        action = action * self.action_bound
        return action, log_prob       


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(QValueNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.out(x)
    

class SACContinuous:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, 
                 device) -> None:
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim)
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim)
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim)
        
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    
    def take_action(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)[0]
        return [action.cpu().item()]


    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) - self.log_alpha.exp() * log_prob
        td_target = rewards + self.gamma * next_value * (1-dones)
        return td_target
    
    def soft_update(self, net, target_net):
        for param, param_target in zip(net.parameters(), target_net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def updaate(self, transition_dict):
        
