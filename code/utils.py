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

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from model import *


def config():
	"""build configruation dictionary"""
	parser = argparse.ArgumentParser(description='')

	#data
	parser.add_argument('--outf', type=str, required=True, help='output folder')
	parser.add_argument('--dataRootTrain', dest='dataRootTrain', type=str, required=True, help='data root to training data')
	parser.add_argument('--dataRootTest', dest='dataRootTest', type=str, required=True, help='data root to test data')
	parser.add_argument('--loadSizeTrain', dest='loadSizeTrain', type=int, default=1000000, help='the amount of trianing data')

	#model
	parser.add_argument('--normName', dest='normName', type=str, default='batch', help='type of normaliazation')
	parser.add_argument('--D-style', dest='D_style', type=str, default='vanilla', help='type of discriminator')
	parser.add_argument('--batchSize', dest='batchSize', type=int, default=64, help='# images in batch')
	parser.add_argument('--workers', dest='workers', type=int, default=4, help='# workers')
	parser.add_argument('--paddedSize', dest='paddedSize', type=int, default=None, help='the final mesh size of input')
	parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of g filters in first conv layer')
	parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of d filters in first conv layer')
	parser.add_argument('--incG', dest='incG', type=int, default=3, help='# of input image channels')
	parser.add_argument('--oncG', dest='oncG', type=int, default=1, help='# of output image channels')
	parser.add_argument('--incD', dest='incD', type=int, default=4, help='# of output image channels')
	parser.add_argument('--downSampleK', dest='downSampleK', type=int, default=5, help='kernel size of downsample conv')
	parser.add_argument('--downSamplePad', dest='downSamplePad', type=int, default=2, help='pad size pf downsample conv')
	parser.add_argument('--reconLoss', dest='reconLoss', type=str, default='l2', help='reconstruction loss name l1, l2, smooth l1...')
	parser.add_argument('--GANLoss', dest='GANLoss', type=str, default='vanilla', help='vanilla mse ...')
	parser.add_argument('--last_activate', dest='last_activate', default=None, help='the final activate function of generator')

	#hyperparameters
	parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='lr of all model')
	parser.add_argument('--Glr', dest='Glr', type=float, default=0.001, help='lr of G')
	parser.add_argument('--Dlr', dest='Dlr', type=float, default=0.00001, help='lr of D')
	parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1, help='weight on reconstruction term in objective')
	parser.add_argument('--gan_lambda', dest='gan_lambda', type=float, default=0.001, help='weight on reconstruction term in objective')
	parser.add_argument('--dist_lambda', dest='dist_lambda', type=float, default=1, help='weight on distribution loss')
	parser.add_argument('--max_stress_lambda', dest='max_stress_lambda', type=float, default=1, help='weight on max stress lambda')
	parser.add_argument('--meshSize', dest='meshSize', type=int, default=128, help='mesh size')
	parser.add_argument('--nepoch', dest='nepoch', type=int, default=4000, help='# of epoch')

	#training
	parser.add_argument('--niter', dest='niter', type=int, default=500, help='# of iter at starting learning rate')
	parser.add_argument('--eval_epoch', type=int, default=5, help="epoch step to eval")
	parser.add_argument('--model_saving_epoch', type=int, default=50, help="model saving epoch step")
	parser.add_argument('--amp', dest='amp', type=int, default=1000, help='the amplification factor of stress map')

	# restore model
	parser.add_argument('--Gmodel', type=str, default=' ', help='model path')
	parser.add_argument('--Dmodel', type=str, default=' ', help='model path')
	parser.add_argument('--Goptim', type=str, default=' ', help='optim path')
	parser.add_argument('--Doptim', type=str, default=' ', help='optim path')
	parser.add_argument('--optim', type=str, default=' ', help='optim path')
	parser.add_argument('--Glrschl', type=str, default=' ', help='lr_scheduler path')
	parser.add_argument('--Dlrschl', type=str, default=' ', help='lr_scheduler path')
	parser.add_argument('--lrschl', type=str, default=' ', help='lr_scheduler path')

	#stage
	parser.add_argument('--eval', action='store_true', help="train mode or evaluation mode")
	parser.add_argument('--ignore_zero', action='store_false', help="ignore there are all zero samples in test data or not")

	args = parser.parse_args()
	return args


class TrainUtils():
	"""This class contains the helper functions

	Attributes:
		opt: configurations
		model_dir: path to model
		summary_dir: path to summary
		summar_writter: tensorboard writer
		loss_name_per_epoch_train: names of metrics in training
		loss_name_val: names of metrics in eval
		metric_name: test matrics
	"""

	def __init__(self, opt):
		""" Initialize the class"""
		self.opt = opt
		self.model_dir = opt.outf+'/model'
		self.summary_dir = opt.outf+'/summary'
		self.summary_writter = SummaryWriter(self.summary_dir)
		self.global_step = 0
		self.epoch = 0
		self.loss_name_per_epoch_train = ['loss_fake_D', 'loss_real', 'loss_D', 'loss_fake_G', 
											'loss_recon', 'loss_G', 'loss_dist', 'loss_max_stress', 
											'MAE', 'MSE']
		self.loss_name_val = ['loss_fake_D', 'loss_real', 'loss_fake_G', 
								'loss_recon', 'loss_dist', 'loss_max_stress']
		self.metric_name = ['MSE', 'MAE', 'PMAE', 'PAE', 'PPAE', 'real_cls', 'fake_cls']
		self.blue = lambda x: '\033[94m' + x + '\033[0m'

		self.build_folders()
		self.save_config()
		self.train_init()
		self.val_init()

	def build_folders(self):
		"""Build directories"""
		try:
			os.makedirs(self.opt.outf, exist_ok=True)
			os.makedirs(self.model_dir, exist_ok=True)
			os.makedirs(self.summary_dir, exist_ok=True)
		except OSError:
			import traceback
			traceback.print_exc()
			pass

	def save_config(self):
		"""Save configuration of experiments"""
		if self.opt.eval:
			ffile = self.opt.outf+'/eval_config.json'
		else:
			ffile = self.opt.outf+'/config.json'

		with open(ffile, 'w') as f: 
			json.dump(vars(self.opt), f, indent=4)

	def train_init(self):
		"""initilize the dictionary for recording metrics in training"""
		self.loss_per_epoch_train = {name:[] for name in self.loss_name_per_epoch_train}

	def val_init(self):
		"""initilize the dictionary for recording metrics in evaluaton"""
		self.loss_val = {name:[] for name in self.loss_name_val}
		self.metrics = {name:[] for name in self.metric_name}

	def update_per_batch_train(self, **kwargs):
		"""record the metric values in training"""
		for k, v in kwargs.items():
			self.loss_per_epoch_train[k].append(v)

	def update_per_batch_val(self, **kwargs):
		"""record the metric values in evluation"""
		for k, v in kwargs.items():
			self.loss_val[k].append(v)

	def update_metrics_val(self, stress_fake, stress_real, cls_fake_D=None, cls_real_D=None):
		"""record the metrics in evaluation"""
		stress_fake = np.reshape(stress_fake, (stress_real.shape[0], -1))
		stress_real = np.reshape(stress_real, (stress_real.shape[0], -1))

		if isinstance(stress_fake, torch.Tensor) or isinstance(stress_real, torch.Tensor):
			stress_fake = stress_fake.numpy()
			stress_real = stress_real.numpy()

		self.metrics['PMAE'].append(TrainUtils.PMAE(stress_fake, stress_real))
		self.metrics['MSE'].append(TrainUtils.MSE(stress_fake, stress_real))
		self.metrics['MAE'].append(TrainUtils.MAE(stress_fake, stress_real))
		self.metrics['PPAE'].append(TrainUtils.PPAE(stress_fake, stress_real))
		self.metrics['PAE'].append(TrainUtils.PAE(stress_fake, stress_real))

		if cls_fake_D is not None and cls_real_D is not None:
			self.metrics['real_cls'].append(np.average(cls_real_D.numpy()))
			self.metrics['fake_cls'].append(np.average(cls_fake_D.numpy()))

	def MSE_MAE(self, stress_fake, stress_real):
		"""calculate MSE and MAE given the stresses"""
		stress_fake = np.reshape(stress_fake, (self.opt.batchSize, -1))
		stress_real = np.reshape(stress_real, (self.opt.batchSize, -1))

		return TrainUtils.MSE(stress_fake, stress_real), TrainUtils.MAE(stress_fake, stress_real)

	def update_histogram_param(self, net):
		"""add weights into summary"""
		assert isinstance(net, dict), 'NET SHOULD BE A DICT'
		for net_name, network in net.items():
			for name, param in network.named_parameters():
				self.summary_writter.add_histogram('%s/%s'%(net_name, name), param, self.global_step)

	def update_summary_train(self):
		"""add calculated metrics into summary in training"""
		for k, v in self.loss_per_epoch_train.items():
			if len(v) != 0:
				self.summary_writter.add_scalar('train/%s'%k, sum(v)/len(v), self.global_step)

	def update_summary_eval(self):
		"""add calculated metrics into summary in evaluation"""
		for k, v in self.loss_val.items():
			if len(v) != 0:
				self.summary_writter.add_scalar('val/%s'%k, sum(v)/len(v), self.global_step)

		for k, v in self.metrics.items():
			if len(v) != 0:
				self.summary_writter.add_scalar('val/%s'%k, sum(v)/len(v), self.global_step)

	def global_step_(self):
		"""update global step"""
		self.global_step += 1

	def epoch_step(self):
		"""update epoch"""
		self.epoch += 1

	def get_model(self):
		"""get model based on configurations"""
		opt = self.opt
		G = Generator(opt.incG, opt.oncG, opt.meshSize, ngf=opt.ngf, last_activate=opt.last_activate, pool=None, 
			norm_layer=opt.normName, use_dropout=False, use_bias=True, k=opt.downSampleK, pad=opt.downSamplePad)
		D = VanillaDiscriminator(opt.incD, opt.meshSize, ndf=opt.ndf, norm_layer=opt.normName, 
			k=opt.downSampleK, pad=opt.downSamplePad)

		print(G)
		print(D)

		if opt.Gmodel != ' ':
			G.load_state_dict(torch.load(opt.Gmodel), strict=False)
			print(opt.Gmodel + ' LOADED SUCCESSFULLY')

		if opt.Dmodel != ' ':
			D.load_state_dict(torch.load(opt.Dmodel), strict=False)
			print(opt.Dmodel + ' LOADED SUCCESSFULLY')

		return G, D

	def get_optim(self, G, D, single=False):
		"""This function returnS two optimizers for G and D respectively or one optimizer for the given models."""
		opt = self.opt
		if not single:
			optimizer_G = optim.Adam(G.parameters(), lr=opt.Glr, betas=(0.9, 0.999))
			optimizer_D = optim.Adam(D.parameters(), lr=opt.Dlr, betas=(0.9, 0.999))

			if opt.Goptim != ' ':
				optimizer_G.load_state_dict(torch.load(opt.Goptim))
				print(opt.Goptim + ' LOADED SUCCESSFULLY')

			if opt.Doptim != ' ':
				optimizer_D.load_state_dict(torch.load(opt.Doptim))
				print(opt.Doptim + ' LOADED SUCCESSFULLY')

			return optimizer_G, optimizer_D
		else:
			if G is not None and D is not None:
				param_list = list(G.parameters())+list(D.parameters())
			elif G is None:
				param_list = G.parameters()
			elif D is None:
				param_list = D.parameters()

			optimizer = optim.Adam(param_list, lr=opt.lr, betas=(0.9, 0.999))

			if opt.optim != ' ':
					optimizer.load_state_dict(torch.load(opt.optim))
					print(opt.optim + ' LOADED SUCCESSFULLY')

			return optimizer

	def get_schl(self, optimizer_G, optimizer_D, optimizer=None, single=False):
		"""This function returnS two schedulers for G and D respectively or one scheduler for the given models."""
		opt = self.opt
		if not single:
			scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.niter, gamma=0.5)
			scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=opt.niter, gamma=0.5)

			if opt.Glrschl != ' ':
				scheduler_G.load_state_dict(torch.load(opt.Glrschl))
				print(opt.Glrschl + ' LOADED SUCCESSFULLY')

			if opt.Dlrschl != ' ':
				scheduler_D.load_state_dict(torch.load(opt.Dlrschl))
				print(opt.Dlrschl + ' LOADED SUCCESSFULLY')

			return scheduler_G, scheduler_D
		else:
			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.niter, gamma=0.5)

			if opt.lrschl != ' ':
				scheduler.load_state_dict(torch.load(opt.lrschl))
				print(opt.lrschl + ' LOADED SUCCESSFULLY')

			return scheduler

	def save_model(self, **models):
		"""save models by their name and global steps"""
		for k,v in models.items():
			torch.save(v, '%s/%s_%d.pth' % (self.model_dir, k, self.global_step))

	def print_training_results(self):
		"""print the losses in training"""
		print('[%d] %s'%(self.epoch, self.blue('train')), end=' ')
		for k, v in self.loss_per_epoch_train.items():
			if len(v) != 0:
				print(k, (sum(v)/len(v)), end=' ')
		print('\n')

	def print_val_results(self):
		"""print the losses and metrics in eval"""
		print('[%d] %s'%(self.epoch, self.blue('val')), end=' ')
		for k, v in self.loss_val.items():
			if len(v) != 0:			
				print(k, (sum(v)/len(v)), end=' ')
		for k, v in self.metrics.items():
			if len(v) != 0:
				print(k, (sum(v)/len(v)), end=' ')			
		print('\n')

	def get_stress_field(self, dists, max_stresses):
		stress_field = [dist*max_stress for dist, max_stress in zip(dists, max_stresses)]
		return np.array(stress_field)

	@staticmethod
	def set_requires_grad(model, requires_grad):
		"""set the model to training mode"""
		for param in model.parameters():
			param.requires_grad = requires_grad

		return model

	@staticmethod
	def MAE(a, b):
		"""a: N*(h*w) or (h*w)"""
		return mean_absolute_error(a, b)

	@staticmethod
	def PMAE(a, b):
		min_b = np.abs(np.amin(b, axis=-1))
		max_b = np.abs(np.amax(b, axis=-1))
		range_ = max_b-min_b
		expand_max_b = np.tile(np.expand_dims(range_, axis=-1), (1, np.shape(b)[-1]))
		new_a = np.divide(a, expand_max_b)
		new_b = np.divide(b, expand_max_b)
		pmae = mean_absolute_error(new_a, new_b)

		return pmae

	@staticmethod
	def MSE(a, b):
		return mean_squared_error(a, b)

	@staticmethod
	def PAE(a, b):
		a_max = np.abs(np.amax(a, axis=-1))
		b_max = np.abs(np.amax(b, axis=-1))
		return mean_absolute_error(a_max, b_max)

	@staticmethod
	def PPAE(a, b):
		a_max = np.abs(np.amax(a, axis=-1))
		b_max = np.abs(np.amax(b, axis=-1))
		new_a_max = np.divide(a_max, b_max)
		return mean_absolute_error(new_a_max, np.ones(np.shape(new_a_max)))


if __name__ == '__main__':
	pass























