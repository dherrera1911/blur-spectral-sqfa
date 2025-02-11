##############
#### IMPORT PACKAGES
##############

import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

filters = torch.load('./data/filters_trained_all.pt')

n_filt = 4
fig, ax = plt.subplots(n_filt, 1, figsize=(16, 10))
for i in range(n_filt):
    ax[i].plot(filters[i][0], 'r')
    ax[i].plot(filters[i][1], 'b')
plt.show()

loss = torch.load('./data/loss_all.pt')

for i in range(n_filt):
    plt.plot(loss[i])
plt.show


