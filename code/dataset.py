"""
Created in 2019
@Author: Haoliang Jiang
@Contact: github.com/haoliangjiang
@Purpose: Code implementation for StressGAN
@Version: python 3.5
@Status: in progress
"""


from __future__ import print_function

import os
import os.path
import sys
import json
import pdb
import random
import math
import copy
import pickle

import torch.utils.data as data
import torch
import numpy as np
import pandas
import matplotlib
# matplotlib.use('TkAgg') #cannot show
# matplotlib.use('agg')
import matplotlib.pyplot as plt

class StressDataset(data.Dataset):
	"""The class for dataloader

	Attributes:
		data_root: path to data
		split: 'train' or 'test'
		padded_size: tensorboard writer
		dataset_size: the size for training data
		condition_nc: the numebr of channels for condtions
		amp: the amplification factor
		eval: split training data to training and eval data
		data: total data
		eval_table: look up table for evaluation
		train_table: look up table for training
	"""

	def __init__(self,
				 data_root,
				 split='train',
				 padded_size=None,
				 dataset_size=None,
				 condition_nc=3,
				 amp=1,
				 ignore_zero=False,
				 eval=False):

		self.data_root = data_root
		self.split = split
		self.padded_size = padded_size
		self.condition_nc = condition_nc
		self.dataset_size = dataset_size
		self.amp = amp
		self.ignore_zero = ignore_zero
		self.eval = eval

		data = np.load(self.data_root).astype(np.float32)

		if self.dataset_size is not None:
			data = data[:self.dataset_size]

		self.mesh_size = int(math.sqrt(np.shape(data)[1]//(self.condition_nc+1)))
		print('MESH SIZE: %d'%self.mesh_size)

		if split == 'train':
			self.data = [(sample[:self.mesh_size**2*self.condition_nc].reshape((self.mesh_size, self.mesh_size, self.condition_nc), order='F').transpose(1,0,2), 
						sample[self.mesh_size**2*self.condition_nc:].reshape((self.mesh_size, self.mesh_size))) for sample in data]
		else:
			if not self.ignore_zero:
				self.data = [(sample[:self.mesh_size**2*self.condition_nc].reshape((self.mesh_size, self.mesh_size, self.condition_nc), order='F').transpose(1,0,2), 
							sample[self.mesh_size**2*self.condition_nc:].reshape((self.mesh_size, self.mesh_size))) for sample in data]
			else:
				self.data = [(sample[:self.mesh_size**2*self.condition_nc].reshape((self.mesh_size, self.mesh_size, self.condition_nc), order='F').transpose(1,0,2), 
							sample[self.mesh_size**2*self.condition_nc:].reshape((self.mesh_size, self.mesh_size))) for sample in data  
							 if np.sum(sample[self.mesh_size**2*self.condition_nc:])!=0]

		if self.padded_size is not None:
			assert self.padded_size%2 == 0, 'ODD DATA PADDING SIZE!'

		print('TOTAL DATA NUM %d'%len(self.data))

		if self.eval:
			num_eval = math.floor(len(self.data)*0.1)
			order = np.random.choice(len(self.data), len(self.data), replace=False)
			self.eval_table = order[:num_eval]
			self.train_table = order[num_eval:]

	def __getitem__(self, index):
		if self.eval:
			condition, real_stress = self.data[self.train_table[index]]
		else:
			condition, real_stress = self.data[index]
		max_stress = 0 #np.amax(real_stress)

		condition_tensor = torch.from_numpy(condition.astype(np.float32)).permute(2, 0, 1)
		stress_tensor = torch.from_numpy(np.array([real_stress*self.amp]).astype(np.float32))
		max_stress = torch.from_numpy(np.array([max_stress]).astype(np.float32))

		return condition_tensor, stress_tensor, max_stress

	def __len__(self):
		if self.eval:
			return len(self.train_table)
		else:
			return len(self.data)

	def get_eval_item(self):
		"""Serves as the getitem function for fine-tuning"""
		condition, real_stress = self.data[self.eval_table(self.eval_index)]
		max_stress = 0

		condition_tensor = torch.from_numpy(condition.astype(np.float32)).permute(2, 0, 1)
		stress_tensor = torch.from_numpy(np.array([real_stress*self.amp]).astype(np.float32))
		max_stress = torch.from_numpy(np.array([max_stress]).astype(np.float32))

		return condition_tensor, stress_tensor, max_stress

	def init_eval(self):
		"""Initialize the index for evaluation"""
		self.eval_index = 0

	def eval_step(self):
		"""One step further"""
		self.eval_index += 1

	def pad(self, nc, sample):
		"""Pad the configurations to a certain size"""
		if nc == 1:
			container = np.zeros((self.padded_size, self.padded_size)).astype(np.float32)
			container[:sample.shape[0],:sample.shape[1]] = sample
		else:
			container = np.zeros((self.padded_size, self.padded_size, nc)).astype(np.float32)
			container[:sample.shape[0],:sample.shape[1],:] = sample

		return container

	def visualization(self, ndata, random=False, data_idx=[]):
		"""Visualize the data"""
		import matplotlib.pyplot as plt
		import matplotlib as mpl
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		if len(data_idx) != 0:
			index = data_idx
			ndata = len(data_idx)
		else:
			order = list(range(self.__len__()))
			if random: random.shuffle(order)
			index = order[:ndata]

		plt.figure(figsize=(ndata, self.condition_nc+1))

		for iidx, idx in enumerate(index):
			condition, stress, _ = self.__getitem__(idx)
			condition, stress = condition.data.numpy(), stress.data.numpy()
			print_list = [condition[i].reshape(self.mesh_size,self.mesh_size) 
							for i in range(self.condition_nc)]
			print_list.append(stress.reshape(self.mesh_size,self.mesh_size))

			for j, data in enumerate(print_list):
				ax = plt.subplot(ndata, self.condition_nc+1, iidx*(self.condition_nc+1)+j+1)
				data_stress = data
				im_stress =ax.imshow(data_stress,cmap='jet',interpolation='nearest')
				stress_min = np.min(data_stress)
				stress_max = np.max(data_stress)
				plt.xticks(())
				plt.yticks(())
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.10)
			cb = plt.colorbar(im_stress, cax=cax, norm=mpl.colors.Normalize(vmin=stress_min, vmax=stress_max))
			cb.ax.tick_params(labelsize='10')

		plt.show()


if __name__ == '__main__':
	pass



