import numpy as np

class MyThompsonSampling:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.lever_pulled = -1
        self.max_reward = 1

        self.FS = [np.zeros(num_arms), np.zeros(num_arms)]
        self.theta = np.zeros(num_arms)

        self.times_pulled = np.zeros(num_arms)
        self.totals = np.zeros(num_arms)
        self.means = np.zeros(num_arms)

    def pull_arm(self) -> int:
        for arm in range(self.num_arms):
            self.theta[arm] = np.random.beta(
                self.FS[1][arm]+1, self.FS[0][arm]+1
            )
            
        self.lever_pulled = np.argmax(self.theta)
        self.times_pulled[self.lever_pulled] += 1
        return self.lever_pulled

    def update_model(self, reward):
        self.totals[self.lever_pulled] += reward;
        self.means[self.lever_pulled] = (
            self.totals[self.lever_pulled] 
            / 
            self.times_pulled[self.lever_pulled]
        )

        if (reward > self.max_reward):
            self.max_reward = reward
            
        reward = reward / self.max_reward 
        res = np.random.binomial(1, reward)
        self.FS[res][self.lever_pulled] += 1