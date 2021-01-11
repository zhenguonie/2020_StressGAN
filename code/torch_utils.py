"""
Created in 2019
@Author: Haoliang Jiang
@Contact: github.com/haoliangjiang
@Purpose: Code implementation for StressGAN
@Version: python 3.5
@Status: in progress
"""


from __future__ import print_function

import math

import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
import matplotlib
# matplotlib.use('Qt4Agg') #cannot show
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import *


def GAN_loss(x, real, loss_fn):
	"""Return the loss based on the loss function of GAN
	Args:
		x->tensor, the predicted tensor
		real->bool, the ground truth
	"""
	if real:
		target = torch.ones(x.size(), dtype=torch.float).cuda()
	else:
		target = torch.zeros(x.size(), dtype=torch.float).cuda()

	loss = loss_fn(x, target)

	return loss

def get_recon_loss(loss_type='l2'):
	"""Return L1 or L2 loss"""
	assert loss_type in ['l2', 'l1'], 'LOSS TYPE NOT SUPPORTED'

	if loss_type == 'l2':
		loss = nn.MSELoss()
	elif loss_type == 'l1':
		loss = nn.L1Loss()

	return loss

def get_GAN_loss(loss_type='vanilla'):
	"""Return the required loss for GAN"""
	assert loss_type in ['mse', 'vanilla'], 'LOSS TYPE NOT SUPPORTED'

	if loss_type == 'mse':
		loss = nn.MSELoss()
	elif loss_type == 'vanilla':
		loss = nn.BCEWithLogitsLoss()

	return loss

def weights_visualization(opt):
	"""To check how weights work, generally tensorboard is better way to check it."""
	train_util = TrainUtils(opt)
	G, D = train_util.get_model()
	G.cuda()
	D.cuda()
	para = []
	for w in G.parameters():
		para += list(w.data.cpu().numpy().reshape(-1))
	print(np.linalg.norm(para,1), np.linalg.norm(para,2))
	_ = plt.hist(para, bins='auto')  # arguments are passed to np.histogram
	plt.title("Histogram with 'auto' bins")
	plt.show()

if __name__ == '__main__':
	args = config()





