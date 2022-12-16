import numpy as np
import matplotlib.pyplot as plt
from lysb.models.LinUCB import *

num_of_bandits = 3
num_of_users = 5

bandit_contexts = np.random.rand(num_of_bandits,10)*3-1
user_contexts = np.random.rand(num_of_users,10)


linucb_experiment = linucb_test_experiment_manager(user_contexts=user_contexts, bandit_contexts=bandit_contexts, num_iterations=1000, alpha=1)

linucb_experiment.experiment()

linucb_experiment.plot()