import numpy as np

import torch
import torch.nn as nn

class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=784
            ),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, X):
        return self.gen(X)