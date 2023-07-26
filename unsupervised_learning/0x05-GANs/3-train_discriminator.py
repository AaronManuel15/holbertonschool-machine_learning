#!/usr/bin/env python3
"""Task 3. Train Discriminator
THIS DEFINITELY NEEDS TO BE REFACTORED. OUTPUT IS NOT BETWEEEN 0 AND 1"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
realSample_Z = __import__('2-sample_Z').realSample_Z

def train_dis(Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer,
              crit):
    """Trains the discriminator
    Args:
        Gen: is the generator model
        Dis: is the discriminator model
        dInputSize: is the dimensionality of the discriminator input
        gInputSize: is the dimensionality of the generator input
        mbatchSize: is the batch size
        steps: is the number of discriminator iterations
        optimizer: is an stochastic gradient descent optimizer object
        crit: is the discriminator BCEloss function
    Returns:
        lossFake: the discriminator loss
        lossReal: the discriminator loss
        FakeData: the generated sample
        RealData: the real sample"""

    for _ in range(steps):
        # resetting the gradients for each iteration
        Dis.zero_grad()

        # Generate noise and fake data labels which is confusing since it is
        # one number??? Randomly throwing one sample at a time until it thinks
        # it was a truly random sample??
        # edit: I think we throw it one real one for shits and giggles
        noise = realSample_Z(0, 1, 'D', (mbatchSize, dInputSize))
        fake_data_labels = torch.ones(mbatchSize, 1)

        # Generate fake data using the Generator on a decent sample size and
        # matching labels
        fakers = realSample_Z(0, 1, 'G', (mbatchSize, gInputSize))
        fakers_output = Gen(fakers)
        fakers_output_labels = torch.zeros(mbatchSize, 1)

        # bringing data and labels together
        data = torch.cat((noise, fakers_output))
        data_labels = torch.cat((fake_data_labels, fakers_output_labels))

        # Train the discriminator by feeding it the data and then calculating
        # loss and backpropagating
        trained_data = Dis(data)
        loss = crit(trained_data, data_labels)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return crit(trained_data, data_labels), data
