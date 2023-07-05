#!/usr/bin/env python3
"""Task 2. Sample Z"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def sample_Z(mu, sigma, sampleType):
    """Creates input for the generator and discriminator
    Args:
        mu: is the mean of the normal distribution
        sigma: is the standard deviation of the normal distribution
        sampleType: is the sample type G or D
    Returns:
        Z: a torch.tensor containing the input for the generator or
        discriminator"""

    return realSample_Z(mu, sigma, sampleType, (1, 50))


def realSample_Z(mu, sigma, sampleType, size):
    """Does the actual work since we need size
    THIS MIGHT NEED NORMALIZED IN THE D CASE STILL"""

    if sampleType == 'G':
        Z = torch.normal(mu, sigma, size=size)
    elif sampleType == 'D':
        size = (1, 1)
        Z = torch.randn(size)
    else:
        return 0
    return Z
