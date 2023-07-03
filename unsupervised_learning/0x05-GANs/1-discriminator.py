#!/usr/bin/env python3
"""Task 1. Initialize Discriminator"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""Task 1. Create Discriminator"""


class Discriminator(nn.Module):
    """Discriminator for the GAN"""
    def __init__(self, input_size, hidden_size, output_size):
        """Constructor for the Discriminator
        Args:
            input: is the dimensionality of the input sample
            hidden_size: is the number of hidden units in the GAN
            output_size: is the dimensionality of the output sample"""
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.main = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
            nn.Linear(self.output_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
