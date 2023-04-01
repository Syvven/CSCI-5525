################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

from hw4_utils import load_MNIST

np.random.seed(2023)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

generator = MyGenerator()
discriminator = MyDiscriminator()

criterion = nn.BCELoss()
epochs = 100

# I did all the stuff here 
#  because I didn't want to make another
#  class to contain all that

# Epoch loop
epochs_gen_loss = []
epochs_discrim_loss = []

epoch_gen_images = []
break_count = 0
max_break_count = 10
min_loss = 9e15
print("--------Training Model--------")
for i in range(epochs):
    total_discrim_loss = 0
    total_gen_loss = 0
    
    # Mini batch loop
    for _,(images,_) in enumerate(train_loader):
        #################################################
        # Discriminator Training
        #################################################

        real_images = images.view(-1, 784)
        real_labels = torch.ones(real_images.shape[0])

        # forward pass the latent vectors through the generator

        fake_images = generator(torch.randn(real_images.shape[0], 128))
        fake_labels = torch.zeros(fake_images.shape[0])

        # pass through real images
        real_outs = discriminator(real_images).view(-1)
        loss_real = criterion(real_outs, real_labels)

        # pass through fake images
        fake_outs = discriminator(fake_images).view(-1)
        loss_fake = criterion(fake_outs, fake_labels)

        # accumulate total loss between the two
        loss = loss_real + loss_fake

        total_discrim_loss += loss.item()

        # backwards the loss and do grad descent

        # I don't exactly know why but it wasn't training properly
        #  when the zero_grad was after the step so I had to put it before 
        #  for it to work
        discriminator.optimizer.zero_grad()
        loss.backward()
        discriminator.optimizer.step()

        #################################################
        # Generator Training
        #################################################
        
        # generate fake images and pass them through discrim
        fake_images = generator(torch.randn(real_images.shape[0], 128))
        fake_labels = torch.ones(fake_images.shape[0])
        fake_outs = discriminator(fake_images).view(-1)

        # get the loss between outputs and the labels we want (ones)
        loss = criterion(fake_outs, fake_labels)

        total_gen_loss += loss.item()

        # do backwards pass

        # I don't exactly know why but it wasn't training properly
        #  when the zero_grad was after the step so I had to put it before 
        #  for it to work
        generator.optimizer.zero_grad()
        loss.backward()
        generator.optimizer.step()

    # save best model
    if (total_gen_loss < min_loss): 
        min_loss = total_gen_loss
        break_count = 0
        torch.save(generator.state_dict(), "model_saves/generator_best.pt")

    # print out stats for the loop
    epochs_discrim_loss.append(total_discrim_loss)
    epochs_gen_loss.append(total_gen_loss)
    print(f"[{i}] Generator Loss: {total_gen_loss}, Discriminator Loss: {total_discrim_loss}")

    # break and load best model if overfitting
    if (total_gen_loss >= min_loss):
        break_count += 1
        if (break_count == max_break_count):
            generator.load_state_dict(
                torch.load("model_saves/generator_best.pt")
            )
            break

    latent_vectors = torch.randn(5, 128)
    epoch_gen_images.append((generator(latent_vectors), i))

plt.plot(epochs_discrim_loss, label = "Discriminator Loss")
plt.plot(epochs_gen_loss, label = "Generator Loss")
plt.legend()
plt.show()

rows = 3
columns = 2

latent_vectors = torch.randn(5, 128)
res = generator(latent_vectors)

fig = plt.figure(figsize=(10, 7))

for i in range(5):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(res[i].view(28, 28).detach().numpy())
    plt.title(f"End Result")

plt.show()

# if it gets through the whole 100 epochs, this will print
#  A LOT of plots. sorry! the instructions said to print
#  after every epoch so...
# realistically, if you want to have it not print out as many,
#  it probably doesn't need to be trained as long to still produce
#  realistic looking results. 

for image_set,epoch in epoch_gen_images:
    fig = plt.figure(figsize=(10, 7))

    # Adds a subplot at the 2nd position
    for i in range(5):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(image_set[i].view(28, 28).detach().numpy())
        plt.title(f"Epoch: {epoch}")

plt.show()

