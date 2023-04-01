import numpy as np

import torch
import torch.nn as nn

class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()

        self.discrim = nn.Sequential(
            nn.Linear(
                in_features=784,
                out_features=1024
            ),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(
                in_features=1024,
                out_features=512
            ),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(
                in_features=512,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(
                in_features=256,
                out_features=1
            ),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, X):
        return self.discrim(X)