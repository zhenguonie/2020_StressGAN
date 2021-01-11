"""
Created in 2019
@Author: Haoliang Jiang
@Contact: github.com/haoliangjiang
@Purpose: Code implementation for StressGAN
@Version: python 3.5
@Status: in progress
"""


from __future__ import print_function

import argparse
import os
import random
import time
import pdb
import json
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
# matplotlib.use('Qt4Agg') #find your own backend
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import *
from torch_utils import GAN_loss, get_recon_loss, get_GAN_loss
from dataset import StressDataset


def train(opt):
	"""
	This function takes care of the training process
	Args:
		opt: the configuration parameters

	"""
	torch.set_printoptions(precision=10)
	opt.manualSeed = random.randint(1, 10000)  # fix seed
	print("RANDOM SEED: ", opt.manualSeed)
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)
	print(opt)

	train_util = TrainUtils(opt)  # initial training util class

	# trining and evaluation dataset and data loader
	D_train = StressDataset(
		data_root=opt.dataRootTrain,
		split='train',
		padded_size=opt.paddedSize,
		dataset_size=opt.loadSizeTrain,
		condition_nc=opt.incG,
		amp=opt.amp)

	D_val = StressDataset(
		data_root=opt.dataRootTest,
		split='val',
		padded_size=opt.paddedSize,
		condition_nc=opt.incG,
		ignore_zero=opt.ignore_zero,
		amp=opt.amp)

	train_loader = torch.utils.data.DataLoader(
				D_train,
				batch_size=opt.batchSize,
				shuffle=True,
				num_workers=int(opt.workers))

	val_loader = torch.utils.data.DataLoader(
				D_val,
				batch_size=opt.batchSize,
				shuffle=False,
				num_workers=int(opt.workers))

	print('DATASET SAMPLES: ', len(D_train), len(D_val))

	# get models, optimizers and schedulers
	G, D = train_util.get_model()

	assert opt.last_activate is  None, 'THERE SHOULD BE NO TANH UPON THE GENERATOR'
	optimizer_G, optimizer_D = train_util.get_optim(G, D)
	scheduler_G, scheduler_D = train_util.get_schl(optimizer_G, optimizer_D)

	recon_loss_fn = get_recon_loss(loss_type=opt.reconLoss)# default l2
	GAN_loss_fn = get_GAN_loss(loss_type=opt.GANLoss)# default vanilla
	lrsch = [scheduler_G, scheduler_D]

	recon_loss_fn = get_recon_loss(loss_type=opt.reconLoss)# default l2

	G.cuda()
	D.cuda()

	num_batch = len(D_train) / opt.batchSize
	time_start = time.time()
	global_step = 0

	for epoch in range(opt.nepoch):

		G = G.train()
		D = D.train()
		data_source_iter = iter(train_loader)
		train_util.train_init()
		train_util.val_init()

		for idx, data in enumerate(data_source_iter, 0):
			condition, stress_real, stress_max = data
			condition, stress_real = condition.cuda(), stress_real.cuda()

			# things related to G
			optimizer_G.zero_grad()
			stress_fake = G(condition)
			loss_recon = recon_loss_fn(stress_fake, stress_real)
			# init optimizer
			D = train_util.set_requires_grad(D, True)  # enable backprop for D
			optimizer_D.zero_grad()
			cls_fake_D = D(torch.cat((condition, stress_fake), dim=1).detach())# this is really important
			cls_real = D(torch.cat((condition, stress_real), dim=1))
			loss_fake_D = GAN_loss(cls_fake_D, False, GAN_loss_fn)
			loss_real = GAN_loss(cls_real, True, GAN_loss_fn)
			loss_D = (loss_fake_D+loss_real)*opt.gan_lambda
			loss_D.backward()           # calculate gradients for D
			optimizer_D.step()          # update D's weights

			# update G
			D = train_util.set_requires_grad(D, False)  # D requires no gradients when optimizing G
			optimizer_G.zero_grad()        # set G's gradients to zero
			cls_fake_G = D(torch.cat((condition, stress_fake), dim=1))
			loss_fake_G = GAN_loss(cls_fake_G, True, GAN_loss_fn)
			loss_G = loss_fake_G*opt.gan_lambda+loss_recon*opt.L1_lambda
			loss_G.backward()                # calculate graidents for G
			optimizer_G.step()             # udpate G's weights

			MSE_batch, MAE_batch = train_util.MSE_MAE(stress_fake.cpu().data, stress_real.cpu().data)
			train_util.update_per_batch_train(loss_fake_D=loss_fake_D.cpu().data, 
												loss_real=loss_real.cpu().data, 
												loss_D=loss_D.cpu().data,
												loss_fake_G=loss_fake_G.cpu().data,
												loss_recon=loss_recon.cpu().data,
												loss_G=loss_G.cpu().data,
												MAE = MAE_batch,
												MSE = MSE_batch)

			train_util.global_step_()

		train_util.update_summary_train()
		train_util.print_training_results()
		train_util.update_histogram_param({'G':G, 'D':D})

		# This is to show the testing curve for each epoch rather than fine tune.
		if epoch % opt.eval_epoch == 0:
			print('START EVALUATION')
			G = G.eval()
			D = D.eval()
			for j, data in enumerate(val_loader, 0):
				condition, stress_real, stress_max = data
				condition, stress_real = condition.cuda(), stress_real.cuda()
				
				stress_fake = G(condition)
				loss_recon = recon_loss_fn(stress_fake, stress_real)
				cls_fake_D = D(torch.cat((condition, stress_fake), dim=1).detach())
				cls_real = D(torch.cat((condition, stress_real), dim=1).detach())
				loss_fake_G = GAN_loss(cls_fake_D, True, GAN_loss_fn)
				loss_fake_D = GAN_loss(cls_fake_D, False, GAN_loss_fn)
				loss_real = GAN_loss(cls_real, True, GAN_loss_fn)
				train_util.update_per_batch_val(loss_fake_D=loss_fake_D.cpu().data, 
												loss_real=loss_real.cpu().data, 
												loss_fake_G=loss_fake_G.cpu().data,
												loss_recon=loss_recon.cpu().data)
				train_util.update_metrics_val(stress_fake.cpu().data, stress_real.cpu().data, 
												cls_fake_D.cpu().data, cls_real.cpu().data)

			train_util.update_summary_eval()
			train_util.print_val_results()

		if epoch % opt.model_saving_epoch == 0:
			train_util.save_model(G=G.state_dict(), 
									D=D.state_dict(),
									G_optim=optimizer_G.state_dict(), 
									D_optim=optimizer_D.state_dict(),
									lr_sch_G=scheduler_G,
									lr_sch_D=scheduler_D)

		for lrs in lrsch: lrs.step()

		train_util.epoch_step()

	train_util.save_model(G=G.state_dict(), D=D.state_dict())


def test(opt):
	"""
	This function takes care of the test process
	Args:
		opt: the configuration parameters

	"""
	blue = lambda x: '\033[94m' + x + '\033[0m'

	train_util = TrainUtils(opt)

	# get dataset and data loader
	D_val = StressDataset(
		data_root=opt.dataRootTest,
		split='val',
		padded_size=opt.paddedSize,
		condition_nc=opt.incG,
		amp=opt.amp,
		ignore_zero=opt.ignore_zero)

	val_loader = torch.utils.data.DataLoader(
			D_val,
			batch_size=opt.batchSize,
			shuffle=False,
			num_workers=int(opt.workers))

	print('DATA SAMPLE: ', len(D_val))

	# restore model
	G, D = train_util.get_model()
	G.cuda()
	D.cuda()

	recon_loss_fn = get_recon_loss(loss_type=opt.reconLoss)# default l2
	GAN_loss_fn = get_GAN_loss(loss_type=opt.GANLoss)# default vanilla

	num_batch = len(val_loader)

	G, D = G.eval(), D.eval()
	train_util.val_init()

	print('START EVALUATION')
	results = []
	gts = []
	conditions = []
	for j, data in enumerate(val_loader, 0):
		condition, stress_real, stress_max = data
		condition, stress_real = condition.cuda(), stress_real.cuda()
		stress_fake = G(condition)
		loss_recon = recon_loss_fn(stress_fake, stress_real)

		cls_fake_D = D(torch.cat((condition, stress_fake), dim=1).detach())
		cls_real = D(torch.cat((condition, stress_real), dim=1).detach())
		loss_fake_G = GAN_loss(cls_fake_D, True, GAN_loss_fn)
		loss_fake_D = GAN_loss(cls_fake_D, False, GAN_loss_fn)
		loss_real = GAN_loss(cls_real, True, GAN_loss_fn)

		train_util.update_per_batch_val(loss_fake_D=loss_fake_D.cpu().data, 
										loss_real=loss_real.cpu().data, 
										loss_fake_G=loss_fake_G.cpu().data,
										loss_recon=loss_recon.cpu().data)
		train_util.update_metrics_val(stress_fake.cpu().data, stress_real.cpu().data, 
										cls_fake_D.cpu().data, cls_real.cpu().data)


	results = np.vstack(results)
	gts = np.vstack(gts)

	np.save(opt.outf+'/results.npy', results)
	np.save(opt.outf+'/gts.npy', gts)

	if opt.ignore_zero:
		train_util.print_val_results()

if __name__ == '__main__':
	args = config()
	if args.eval:
		eval(args)
	else:
		train(args)







