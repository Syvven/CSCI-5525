################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
from matplotlib import pyplot as plt

from Environment import Environment

from MyEpsilonGreedy import MyEpsilonGreedy
from MyUCB import MyUCB
from MyThompsonSampling import MyThompsonSampling

num_arms = 8 # Number of arms for each bandit
num_rounds = 500 # Variable 'T' in the writeup
num_repeats = 10 # Variable 'repetitions' in the writeup

# Gaussian environment parameters
means = [7.2, 20.8, 30.4, 10.3, 40.7, 50.1, 1.5, 45.3]
variances = [0.01, 0.02, 0.03, 0.02, 0.04, 0.001, 0.0007, 0.06]

if len(means) != len(variances):
    raise ValueError('Number of means and variances must be the same.')
if len(means) != num_arms or len(variances) != num_arms:
    raise ValueError('Number of means and variances must be equal to the number of arms.')

# Bernoulli environment parameters
p = [0.45, 0.13, 0.71, 0.63, 0.11, 0.06, 0.84, 0.43]

if len(p) != num_arms:
    raise ValueError('Number of Bernoulli probabily values p must be equal to the number of arms.')

# Epsilon-greedy parameter
epsilon = 0.1

if epsilon < 0:
    raise ValueError('Epsilon must be >= 0.')

gaussian_env_params = {'means':means, 'variances':variances}
bernoulli_env_params = {'p':p}

# Use these two objects to simulate the Gaussian and Bernoulli environments.
# In particular, you need to call get_reward() and pass in the arm pulled to receive a reward from the environment.
# Use the other functions to compute the regret.
# See Environment.py for more details. 
gaussian_env = Environment(name='Gaussian', env_params=gaussian_env_params)
bernoulli_env = Environment(name='Bernoulli', env_params=bernoulli_env_params)

#####################
# ADD YOUR CODE BELOW
#####################

from MyRandomAgent import MyRandomAgent

repetitions = 10
rounds_per = 500

envs = [gaussian_env, bernoulli_env]
# the true best values to calculate regret with
ustars = [
    gaussian_env.get_mean_reward(gaussian_env.get_opt_arm()),
    bernoulli_env.get_mean_reward(bernoulli_env.get_opt_arm())
]
regrets_per_round = np.zeros((repetitions, 4, 2, rounds_per))
for rep in range(repetitions):
    cum_regret = np.zeros((4, 2))
    # reset agents for each new repetition
    agents = [
        [MyRandomAgent(num_arms),            MyRandomAgent(num_arms)           ],
        [MyEpsilonGreedy(num_arms, epsilon), MyEpsilonGreedy(num_arms, epsilon)],
        [MyUCB(num_arms),                    MyUCB(num_arms)                   ],
        [MyThompsonSampling(num_arms),       MyThompsonSampling(num_arms)      ]
    ]

    for round in range(rounds_per):
        # pull each agent for each env
        for i,agent_pair in enumerate(agents):
            lever_gauss = agent_pair[0].pull_arm()
            lever_bern  = agent_pair[1].pull_arm()

            # get reward
            reward_gauss = envs[0].get_reward(lever_gauss)
            reward_bern  = envs[1].get_reward(lever_bern)

            # update model
            agent_pair[0].update_model(reward_gauss)
            agent_pair[1].update_model(reward_bern)

            # accumulate regret
            cum_regret[i][0] += ustars[0] - agent_pair[0].means[lever_gauss]
            cum_regret[i][1] += ustars[1] - agent_pair[1].means[lever_bern]

            # update the totals
            regrets_per_round[rep][i][0][round] += cum_regret[i][0]
            regrets_per_round[rep][i][1][round] += cum_regret[i][1]

# get average
# regrets_per_round /= repetitions

mean_reg = np.zeros((4, 2, rounds_per))
stddev_reg = np.zeros((4, 2, rounds_per))

for i in range(repetitions): mean_reg += regrets_per_round[i]
mean_reg /= repetitions

for i in range(repetitions): stddev_reg += np.power((regrets_per_round[i] - mean_reg), 2)
stddev_reg /= repetitions
stddev_reg = np.sqrt(stddev_reg)

x = np.arange(num_rounds)
fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))

# plot the gaussian agents
ax1.set_title("Agents in Gaussian Environment")
ax1.set_xlabel('Rounds')
ax1.set_ylabel('Cumulative Regret')
ax1.plot(mean_reg[0][0], label="Random Agent")
ax1.fill_between(x, mean_reg[0][0]-stddev_reg[0][0], mean_reg[0][0]+stddev_reg[0][0], alpha=0.15)
ax1.plot(mean_reg[1][0], label="Epsilon Greedy")
ax1.fill_between(x, mean_reg[1][0]-stddev_reg[1][0], mean_reg[1][0]+stddev_reg[1][0], alpha=0.15)
ax1.plot(mean_reg[2][0], label="UCB")
ax1.fill_between(x, mean_reg[2][0]-stddev_reg[2][0], mean_reg[2][0]+stddev_reg[2][0], alpha=0.15)
ax1.plot(mean_reg[3][0], label="Thompson Sampling")
ax1.fill_between(x, mean_reg[3][0]-stddev_reg[3][0], mean_reg[3][0]+stddev_reg[3][0], alpha=0.15)
ax1.legend(loc='upper left')
plt.show()

fig, ax2 = plt.subplots(1, 1, figsize=(16, 10))
# plot the bernoulli agents
ax2.set_title("Agents in Bernoulli Environment")
ax2.set_xlabel('Rounds')
ax2.set_ylabel('Cumulative Regret')
ax2.plot(mean_reg[0][1], label="Random Agent")
ax2.fill_between(x, mean_reg[0][1]-stddev_reg[0][1], mean_reg[0][1]+stddev_reg[0][1], alpha=0.1)
ax2.plot(mean_reg[1][1], label="Epsilon Greedy")
ax2.fill_between(x, mean_reg[1][1]-stddev_reg[1][1], mean_reg[1][1]+stddev_reg[1][1], alpha=0.1)
ax2.plot(mean_reg[2][1], label="UCB")
ax2.fill_between(x, mean_reg[2][1]-stddev_reg[2][1], mean_reg[2][1]+stddev_reg[2][1], alpha=0.1)
ax2.plot(mean_reg[3][1], label="Thompson Sampling")
ax2.fill_between(x, mean_reg[3][1]-stddev_reg[3][1], mean_reg[3][1]+stddev_reg[3][1], alpha=0.1)
ax2.legend(loc='upper left')

plt.show()
