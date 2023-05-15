import numpy as np
np.random.seed(0)

P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

P = np.array(P)
rewards = [-1, -2, -2, 10, 1, 0]


def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i]]
    return G

gamma = 0.5
chain = [0, 1, 2, 5]
start_index=0
print(compute_return(start_index, chain, gamma))



def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num,  states_num) - gamma*P),
                   rewards)
    return value

V = compute(P, rewards, gamma, 6)
print(V)