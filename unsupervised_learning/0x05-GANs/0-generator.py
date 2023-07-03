#!/usr/bin/env python3
"""Task 0. Initialize Generator"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Generator(nn.Module):
    """Generator for the GAN"""
    def __init__(self, input_size, hidden_size, output_size):
        """Constructor for the Generator
        Args:
            input: is the dimensionality of the noise vector
            hidden_size: is the number of hidden units in the GAN
            output_size: is the dimensionality of the generated sample"""
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.main = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Tanh(),
            nn.Linear(self.output_size, self.output_size)
        )

    def forward(self, x):
        """Defines the forward pass for the Generator"""
        return self.main(x)
