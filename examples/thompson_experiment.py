import numpy as np
import matplotlib.pyplot as plt
from lysb.models.Thompson import *

num_of_bandits = 3
num_of_users = 5

bandit_contexts = np.random.rand(num_of_bandits,10)*3-1
user_contexts = np.random.rand(num_of_users,10)


thompson_experiment = thompson_test_experiment_manager(user_contexts=user_contexts, bandit_contexts=bandit_contexts, num_iterations=1000)

thompson_experiment.experiment()

thompson_experiment.plot()