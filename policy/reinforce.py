import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 learning_rate, gamma, device):
        
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                          lr=learning_rate)
        self.gamma = gamma
        self.device = device


    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()

        a_loss = 0
        a = False
        # 这个是非并行的，运行效率慢
        if a:
            for i in reversed(range(len(reward_list))):
                reward = reward_list[i]
                state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
                action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
                log_prob = torch.log(self.policy_net(state).gather(1, action))
                G = self.gamma* G + reward
                loss = -log_prob * G
                a_loss += loss
                loss.backward()
            self.optimizer.step()
        else:
        #这是并行版本
            
            episode_len = len(reward_list)
            rewards = torch.tensor(reward_list, dtype=torch.float).to(self.device)
            
            ones = torch.ones(episode_len, episode_len).to(self.device)
            exp = torch.triu(ones).cumsum_(dim=1) - 1
            # [1, 1, 1], [ 0, 1, 1], [0, 0, 1] ->
            # [ [1, 2, 3], [0, 1, 2], [0, 0, 1]] - 1
            # [ [gamma**0, gamma**1, gamma**2] * [r0, r1, r2] ]
            # [ [gamma**-inf, gamma**0, gamma**1] * [r0, r1, r2] ]
            # [ [gamma**-inf, gamma**0, gamma**0] * [r0, r1, r2] ]
            exp_mask = exp.masked_fill(exp[:, :] == -1, float('inf'))
            gamma_range = gamma ** exp_mask
            g = rewards * gamma_range
            # G0, G1, G2, ... , Gn
            G = g.sum(dim=1)
        
            states = torch.tensor(state_list, dtype=torch.float).to(self.device)
            actions = torch.tensor(action_list).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(states).gather(1, actions))
            loss = G @ -log_prob
            #print('loss', a_loss, loss)
            loss.backward()
            self.optimizer.step()
        



learning_rate = 1e-3
num_episodes = 200
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v0"
env = gym.make(env_name)

torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  device)


return_list = []
for i in range(50):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()[0]
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()