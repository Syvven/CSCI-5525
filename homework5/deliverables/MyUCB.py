import numpy as np

class MyUCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.lever_pulled = -1
        self.count = 0
        self.round = 1

        self.times_pulled = np.zeros(num_arms)
        self.totals = np.zeros(num_arms)
        self.means = np.zeros(num_arms)

    def pull_arm(self) -> int:
        if (self.count == self.num_arms):
            temp_means = self.means + np.sqrt(2 * np.log(self.round) / self.times_pulled)
            self.lever_pulled = np.argmax(temp_means)
        else:
            self.lever_pulled = self.count 
            self.count += 1

        self.round += 1
        self.times_pulled[self.lever_pulled] += 1
        return self.lever_pulled

    def update_model(self, reward):
        self.totals[self.lever_pulled] += reward;
        self.means[self.lever_pulled] = (
            self.totals[self.lever_pulled] 
            / 
            self.times_pulled[self.lever_pulled]
        )
    