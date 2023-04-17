import numpy as np
from typing import Dict

class Environment():

    def __init__(self, name: str, env_params: Dict) -> None:
        self.name = name
        if self.name == 'Gaussian':
            self.means = env_params['means']
            self.vars = env_params['variances']
        elif self.name == 'Bernoulli':
            self.p = env_params['p']
        else:
            raise ValueError("Unknown environment name:", self.name, ". Must use either 'Gaussian' or 'Bernoulli'.")

    # returns a reward when an arm is pulled
    def get_reward(self, arm: int) -> float:
        if self.name == 'Gaussian':
            return np.random.normal(self.means[arm], self.vars[arm])
        elif self.name == 'Bernoulli':
            return np.random.binomial(1, self.p[arm])

    # returns the optimal reward
    def get_opt_reward(self) -> float:
        if self.name == 'Gaussian':
            return max(self.means)
        elif self.name == 'Bernoulli':
            return max(self.p)

    # returns the optimal arm
    def get_opt_arm(self) -> int:
        if self.name == 'Gaussian':
            return np.argmax(self.means)
        elif self.name == 'Bernoulli':
            return np.argmax(self.p)
        
    # returns the mean reward for a given arm
    def get_mean_reward(self, arm: int) -> float:
        if self.name == 'Gaussian':
            return self.means[arm]
        elif self.name == 'Bernoulli':
            return self.p[arm]
