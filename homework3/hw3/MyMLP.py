import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)  
        super(MyMLP, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

        self.lr = learning_rate 
        self.max_epochs = max_epochs
        self.output_size = output_size

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output

        # need to flatten to batchsize x (28*28)
        x = x.view(x.shape[0], -1)
        return self.seq(x)

    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''

        print("--------Training Model--------")
        print(f"Params:")
        print(f"lr = {self.lr}")
        print(f"optim = {optimizer}")
        print("------------------------------")

        # Epoch loop
        epochs_loss = []
        epochs_err = []
        break_count = 0
        max_break_count = 4
        for i in range(self.max_epochs):
            total_loss = 0
            total_err = 0
            total_samples = 0
            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):
                yact = torch.zeros(labels.shape[0], self.output_size)
                ind1 = torch.arange(0, labels.shape[0], dtype=int)
                yact[ind1, labels] = 1

                # Forward pass (consider the recommmended functions in homework writeup)
                yhat = self.forward(images)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                # Track the loss and error rate
                loss = criterion(yhat, yact)
                total_loss += loss.item() 

                for k,pred in enumerate(yhat):
                    ind = np.argmax(pred.detach().numpy())
                    total_err += 1 if labels[k].item() != ind else 0

                total_samples += images.shape[0]

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

            epochs_loss.append(total_loss)
            epochs_err.append(total_err / total_samples)
            print(f"[{i}] Loss: {total_loss}, Error Rate: {epochs_err[-1]}")
            if (len(epochs_loss) >= 2) and (epochs_loss[-1] > epochs_loss[-2]): 
                break_count += 1

            if break_count == max_break_count:
                print("Loss Not Decreasing -- Stopping Early")
                break
            
        # Print/return training loss and error rate in each epoch
        return epochs_loss, epochs_err
            


    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''

        total_loss = 0
        total_err = 0
        total_samples = 0
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                yact = torch.zeros(labels.shape[0], self.output_size)
                ind1 = torch.arange(0, labels.shape[0], dtype=int)
                yact[ind1, labels] = 1
                # Compute prediction output and loss
                yhat = self.forward(images)
                # Measure loss and error rate and record
                loss = criterion(yhat, yact)
                total_loss += loss.item() 

                for k,pred in enumerate(yhat):
                    ind = np.argmax(pred.detach().numpy())
                    total_err += 1 if labels[k].item() != ind else 0

                total_samples += images.shape[0]

        # Print/return test loss and error rate
        return total_loss, total_err/total_samples


