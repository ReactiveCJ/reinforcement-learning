#import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self, ncol, nrow) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = 0

    def step(self, action):
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done
    
    def reset(self):
        self.x = 0
        self.y = 0
        return self.y * self.ncol + self.x
    

class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow*ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma*self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
    
    def update_n(self, s0, a0, r, s1, a1, n, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(n)):
                G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n步Sarsa的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []


class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow*ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma*self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error



def run_agent(env, agent, is_Q=False):

    np.random.seed(0)

    num_episodes = 500
    return_list = []
    #T=50
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' %i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done=False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward
                    if is_Q:
                        agent.update(state, action, reward, next_state)
                    else:
                        agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                return_list.append(episode_return)
           
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
     for i in range(env.nrow):
        for j in range(env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i*env.ncol + j)
                pi_str = ''
                for k in range(len(a)):
                    if a[k] > 0:
                        pi_str += action_meaning[k]
                if len(pi_str) < 4:
                    pi_str += '_' * (4-len(pi_str))
                #for k in range(len(action_meaning)):
                #    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



ncol = 12
nrow = 4
np.random.seed(0)
env = CliffWalkingEnv(ncol, nrow)

epsilon = 0.1
alpha = 0.1
gamma = 0.9
sarsa = Sarsa(ncol, nrow, epsilon, alpha, gamma)
qlearning = QLearning(ncol, nrow, epsilon, alpha, gamma)

action_meaning = ['^', 'v', '<', '>']


run_agent(env, sarsa)
print('Sarsa算法最终收敛得到的策略为：')
print_agent(sarsa, env, action_meaning, list(range(37, 47)), [47])

run_agent(env, qlearning, True)
print('QLearning算法最终收敛得到的策略为：')
print_agent(qlearning, env, action_meaning, list(range(37, 47)), [47])
