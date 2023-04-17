import numpy as np

class MyRandomAgent: 
    def __init__(self, num_arms):
        self.times_pulled = np.zeros(num_arms)
        self.totals = np.zeros(num_arms)
        self.means = np.zeros(num_arms)

        self.num_arms = num_arms
        self.lever_pulled = -1

    def pull_arm(self) -> int:
        self.lever_pulled = np.random.randint(0, self.num_arms)
        self.times_pulled[self.lever_pulled] += 1
        return self.lever_pulled

    def update_model(self, reward):
        self.totals[self.lever_pulled] += reward;
        self.means[self.lever_pulled] = (
            self.totals[self.lever_pulled] 
            / 
            self.times_pulled[self.lever_pulled]
        )