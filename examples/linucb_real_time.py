import numpy as np
import matplotlib.pyplot as plt
from lysb.models.LinUCB import linUCB

num_of_bandits = 3
num_of_users = 5

bandit_contexts = np.random.rand(num_of_bandits,10)*3-1
user_contexts = np.random.rand(num_of_users,10)

linucb = linUCB(num_bandits=num_of_bandits, num_features=10, alpha=1)

for i in range(1000):
    reward_matrix = user_contexts @ bandit_contexts.T
    context_id = np.random.randint(len(user_contexts)) # simulating a live flow of user data
    linucb_res_id = linucb.ask(user_contexts[context_id])
    linucb_reward = reward_matrix[context_id][linucb_res_id]
    linucb.update(linucb_reward)


linucb.plot()