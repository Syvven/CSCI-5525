import numpy as np

class MyEpsilonGreedy:
    def __init__(self, num_arms, epsilon):
        self.times_pulled = np.zeros(num_arms)
        self.totals = np.zeros(num_arms)
        self.means = np.zeros(num_arms)

        self.epsilon = epsilon
        self.num_arms = num_arms
        self.lever_pulled = -1

    def pull_arm(self) -> int:
        v = np.random.uniform(0,1)
        if (v <= self.epsilon): 
            self.lever_pulled = np.random.randint(0, self.num_arms)
        else: 
            self.lever_pulled = np.argmax(self.means)

        self.times_pulled[self.lever_pulled] += 1

        return self.lever_pulled

    def update_model(self, reward):
        self.totals[self.lever_pulled] += reward;
        self.means[self.lever_pulled] = (
            self.totals[self.lever_pulled] 
            / 
            self.times_pulled[self.lever_pulled]
        )
