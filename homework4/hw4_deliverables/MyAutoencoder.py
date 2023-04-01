import numpy as np

import torch
import torch.nn as nn

class MyAutoencoder(nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(
                in_features=784,
                out_features=400
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=400, 
                out_features=2
            ),
            nn.Tanh()
        )

        self.decode = nn.Sequential(
            nn.Linear(
                in_features=2,
                out_features=400
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=400,
                out_features=784
            ),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001
        )
        self.criterion = nn.MSELoss()

    def forward(self, X):
        X_reduced = self.encode(X)
        X_replicated = self.decode(X_reduced)

        return X_reduced, X_replicated

    def fit(self, train_loader, epochs):
        train_set, valid_set = torch.utils.data.random_split(
            train_loader.dataset,
            [0.8, 0.2], 
            generator=torch.Generator().manual_seed(42)
        )

        train_loader_v2 = torch.utils.data.DataLoader(
            train_set, batch_size=32, shuffle=False
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=32, shuffle=False
        )

        print("--------Training Model--------")

        # Epoch loop
        epochs_loss = []
        break_count = 0
        max_break_count = 5
        min_loss = 9e15
        for i in range(epochs):
            total_loss = 0
            # Mini batch loop
            for _,(images,_) in enumerate(train_loader_v2):
                images = images.view(-1, 784)

                # Forward pass (consider the recommmended functions in homework writeup)
                _, images_rep = self.forward(images)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                # Track the loss
                loss = self.criterion(images_rep, images)
                total_loss += loss.item() 

                loss.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()

            # print out stats for the loop
            epochs_loss.append(total_loss)
            print(f"[{i}] Train Loss: {total_loss}")

            # training step done, now validate

            valid_loss = 0
            with torch.no_grad(): # no backprop step so turn off gradients
                for _,(images,_) in enumerate(valid_loader):
                    # Compute prediction output and loss
                    images = images.view(-1, 784)

                    _, images_rep = self.forward(images)
            
                    loss = self.criterion(images_rep, images)
                    valid_loss += loss.item() 

            # save best model
            if (valid_loss < min_loss): 
                min_loss = valid_loss
                break_count = 0
                torch.save(self.state_dict(), "model_saves/autoencoder_best.pt")

            print(f"[{i}] Validation Loss: {valid_loss}, Best Validation Loss: {min_loss}\n")

            # break and load best model if overfitting
            if (valid_loss >= min_loss):
                break_count += 1
                if (break_count == max_break_count):
                    self.load_state_dict(
                        torch.load("model_saves/autoencoder_best.pt")
                    )
                    break
            
        # Print/return training loss and error rate in each epoch
        return epochs_loss

    def transform(self, X):
        return self.forward(X.view(-1, 784))
        