import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


  
class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    




class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VANet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super(VANet, self).__init__()
        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc_1(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1 ,1)
        return Q

class DQN:
    #dqn_type: DQN/DoubleDQN
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, 
                 gamma, epsilon, target_update, device, dqn_type='QDQ'):
        self.action_dim = action_dim

        if dqn_type == 'DuelingDQN':
            self.q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VANet(state_dim, hidden_dim, self.action_dim).to(device)

        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)


        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type
    
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)          
            action = self.q_net(state).argmax().item()
        return action
    
    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) #网络1的Q值
        if self.dqn_type == 'DoubleDQN':
            max_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_actions)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1- dones)

        dqn_loss = F.mse_loss(q_values, q_targets, reduction='mean')

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    
def dis_to_con(dis_action, env, action_dim):
    low_bound = env.action_space.low[0]
    up_bound = env.action_space.high[0]
    return low_bound + dis_action * (up_bound - low_bound)/(action_dim-1)


def train(agent, env, epochs, num_episodes, replay_buffer, minimal_size, batch_size):

    return_list = []
    max_q_value_list = []
    max_q_value = 0
    num_play = 0
    for i in range(epochs):
        with tqdm(total=num_episodes, desc='Iteration %d' % i) as pbar:
            for j in range(num_episodes):
                num_play = 0
                episode_return = 0
                state = env.reset()[0]
                done = False
                while not done:   
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)
                    # action_continuous = dis_to_con(action, env, agent.action_dim)
                    # next_state, reward, done, _, _ = env.step([action_continuous])
                    # if num_play % 1000 == 0:
                    #     done = True
                    next_state, reward, done, _, _ = env.step(action)
                    
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    #print(replay_buffer.size(), done)
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                        }
                        agent.update(transition_dict)
                    return_list.append(episode_return)
                if (j + 1) % 10 == 0:
                    pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes * i + j + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list        
            
lr = 2e-3
epochs=10
num_episodes = 50
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 500
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v1'
#env_name = 'Pendulum-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
#action_dim = 11


agent  = DQN(state_dim, hidden_dim, action_dim, lr, gamma,
        epsilon, target_update, device, dqn_type='DQN')

return_list, max_q_value_list = train(agent, env, epochs, num_episodes, replay_buffer, minimal_size, batch_size)


episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()