#!/usr/bin/env python3
"""Task 4. Train Discriminator
THIS DEFINITELY NEEDS TO BE REFACTORED. OUTPUT IS NOT BETWEEEN 0 AND 1"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
realSample_Z = __import__('2-sample_Z').realSample_Z


def train_gen(Gen, Dis, gInputSize, mbatchSize, steps, optimizer, crit):
    """Trains the generator
    Args:
        Gen: is the generator model
        Dis: is the discriminator model
        gInputSize: is the dimensionality of the generator input
        mbatchSize: is the batch size
        steps: is the number of generator iterations
        optimizer: is an stochastic gradient descent optimizer object
        crit: is the discriminator BCEloss function
    Returns:
        error of fake data set
        fake data set of type torch.tensor"""

    for _ in range(steps):

        # Generate fake data
        fakers = realSample_Z(0, 1, 'G', (mbatchSize, gInputSize))
        fakers_output = Gen(fakers)
        labels = torch.ones((mbatchSize, 1))

        # Train the generator
        # resetting the gradients for each iteration
        Gen.zero_grad()
        trained_data = Dis(fakers_output)
        loss = crit(trained_data, labels)
        loss.backward()
        optimizer.step()

    return loss, fakers
