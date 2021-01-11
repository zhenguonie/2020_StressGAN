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
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim

from torch_utils import GAN_loss

def get_norm_layer(norm_type='instance'):
	"""get the correct norm layer"""

	if norm_type == 'batch':
		norm_layer = nn.BatchNorm2d
		affine = False
		track_running_stats = False
	elif norm_type == 'instance':
		norm_layer = nn.InstanceNorm2d
		affine = False
		track_running_stats = False
	elif norm_type == 'none':
		norm_layer = lambda x: Identity()
		affine = None
		track_running_stats = None
	else:
		raise NotImplementedError('NORMALIZATION LAYER [%s] IS NOT FOUND' % norm_type)
	return norm_layer, affine, track_running_stats

class Identity(nn.Module):
	"""An identity layer"""

	def forward(self, x):
		return x

class Generator(nn.Module):
	"""This class is to define the generator achitecture"""

	def __init__(self, inner_nc, outer_nc, mesh_size, ngf=64, last_activate=None, pool=None, norm_layer='batch', use_dropout=False, use_bias=True, k=4, pad=1):
		super(Generator, self).__init__()
		self.outer_nc = outer_nc
		self.inner_nc = inner_nc
		self.mesh_size = mesh_size
		self.pool = pool
		self.norm_layer = norm_layer
		self.use_dropout = use_dropout
		self.k = k
		self.pad = pad
		self.coarse_mesh = 32
		self.feature_dim = 512

		# if batch norm is used, no need to use add bias to previous layers
		if norm_layer == 'batch':
			self.use_bias = False
		elif norm_layer == 'instance':
			self.use_bias = True

		self.last_activate = last_activate

		if pool is None:
			nlayer = int(math.log(mesh_size, 2))
		else:
			nlayer = int(math.log(self.coarse_mesh, 2))

		# the encoder
		downsample = []

		# the input layer
		downsample.append(nn.Conv2d(inner_nc, ngf, kernel_size=self.k,
							 		stride=2, padding=self.pad, bias=self.use_bias))

		# intermediate layers
		for i in range(0, nlayer-2, 1): # 0-nlayer-3
			in_nc = min(ngf*(2**(i)), self.feature_dim)
			out_nc = min(ngf*(2**(i+1)), self.feature_dim)
			downsample.append(SampleBlock(in_nc, out_nc, self.norm_layer, self.use_bias, 
										'down', k=self.k, pad=self.pad))

		# encode layer
		downsample.append(DownSampleFeature(out_nc, self.feature_dim, self.norm_layer, 
											self.use_bias, k=self.k, pad=self.pad))
		self.downsample = nn.Sequential(*downsample)

		# the decoder
		upsample = []

		upsample.append(UpSampleFeature(512, out_nc, self.norm_layer, self.use_bias, k=self.k, pad=self.pad))

		for j in range(nlayer-3, -1, -1):# nlayer-3-0
			in_nc = min(ngf*(2**(j+1)), self.feature_dim)
			out_nc = min(ngf*(2**(j)), self.feature_dim)
			upsample.append(SampleBlock(in_nc, out_nc, self.norm_layer, self.use_bias, 'up', k=self.k, pad=self.pad))

		if self.k == 5 and self.pad == 2:
			last_mod = [nn.ConvTranspose2d(ngf, outer_nc, kernel_size=self.k, stride=2, 
						padding=self.pad, bias=self.use_bias, output_padding=1)]
		else:
			last_mod = [nn.ReLU(True), 
						nn.ConvTranspose2d(ngf, outer_nc, kernel_size=self.k, stride=2, padding=self.pad)]

		if self.last_activate is not None:
			last_mod += [nn.Tanh()]

		upsample += last_mod
		self.upsample = nn.Sequential(*upsample)

	def forward(self, x):
		feature_raw = self.downsample(x)

		if self.pool is None:
			feature = feature_raw
		elif self.pool == 'average':
			N, C, H, W = feature.size()
			feature = F.avg_pool2d(feature_raw, (H, W))
		elif self.pool == 'max':
			N, C, H, W = feature.size()
			feature = F.max_pool2d(feature_raw, (H, W))

		stress = self.upsample(feature)
		return stress

class SampleBlock(nn.Module):
	"""This class builds the intermideate blocks for the encoder and decoder"""

	def __init__(self, in_nc, out_nc, norm_layer, use_bias, direction, k=4, pad=1):
		super(SampleBlock, self).__init__()
		self.in_nc = in_nc
		self.out_nc = in_nc
		self.norm_layer = norm_layer
		self.use_bias = use_bias
		self.k = k
		self.pad = pad
		assert direction in ['down', 'up'], 'DIRECTION IS UP OR DOWN'

		if direction == 'down':
			self.activate = nn.LeakyReLU(0.2, True)
			self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=self.k, stride=2, 
						padding=self.pad, bias=self.use_bias)
		elif direction == 'up':
			self.activate = nn.ReLU(True)
			self.conv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=self.k, 
						stride=2, padding=self.pad, bias=self.use_bias)

			if self.k == 5 and self.pad == 2:
				self.conv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=self.k, 
							stride=2, padding=self.pad, bias=self.use_bias, output_padding=1)

		norm, affine, track_running_stats = get_norm_layer(norm_type=self.norm_layer)
		self.norm = norm(out_nc, affine=affine, track_running_stats=track_running_stats)

	def forward(self, x):
		x = self.activate(x)
		x = self.conv(x)
		x = self.norm(x)
		return x

class DownSampleFeature(nn.Module):
	"""This class builds the feature encoder layer"""

	def __init__(self, in_nc, out_nc, norm_layer, use_bias, k=4, pad=1):
		super(DownSampleFeature, self).__init__()
		self.in_nc = in_nc
		self.out_nc = in_nc
		self.norm_layer = norm_layer
		self.use_bias = use_bias
		self.k = k
		self.pad = pad
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=self.k, stride=2, padding=self.pad, bias=self.use_bias)
		self.relu = nn.ReLU(True)

	def forward(self, x):
		x = self.lrelu(x)
		x = self.conv(x)
		x = self.lrelu(x)
		return x

class UpSampleFeature(nn.Module):
	"""This class builds the feature decoder layer"""

	def __init__(self, in_nc, out_nc, norm_layer, use_bias, k=4, pad=1):
		super(UpSampleFeature, self).__init__()
		self.in_nc = in_nc
		self.out_nc = in_nc
		self.norm_layer = norm_layer
		self.use_bias = use_bias
		self.k = k
		self.pad = pad
		self.conv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=self.k, stride=2, 
					padding=self.pad, bias=self.use_bias)

		if self.k == 5 and self.pad == 2:
			self.conv = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=self.k, stride=2, 
						padding=self.pad, bias=self.use_bias, output_padding=1)

		norm, affine, track_running_stats = get_norm_layer(norm_type=self.norm_layer)
		self.norm = norm(out_nc, affine=affine, track_running_stats=track_running_stats)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		return x


class VanillaDiscriminator(nn.Module):
	"""Defines a VanillaDiscriminator discriminator"""

	def __init__(self, input_nc, mesh_size, ndf=64, pool=None, norm_layer='batch', k=4, pad=1):
		"""Construct a Discriminator->Nx512x1->linear->Nx1
		Args:
			the model will be a size-invariant model when pool is not None. In this case, nlayer=5 which means mesh_size>=32
		"""
		super(VanillaDiscriminator, self).__init__()
		if norm_layer == 'batch':
			self.use_bias = False
		else:
			self.use_bias = True

		self.pool = pool
		self.k = k
		self.pad = pad
		self.norm_layer = norm_layer
		self.feature_dim = 512

		if pool is None:
			nlayer = int(math.log(mesh_size, 2))
		else:
			nlayer = int(math.log(32, 2))

		norm, affine, track_running_stats = get_norm_layer(norm_type=self.norm_layer)

		model = [nn.Conv2d(input_nc, ndf, kernel_size=self.k, stride=2, padding=self.pad), 
				norm(ndf, affine=affine, track_running_stats=track_running_stats), nn.LeakyReLU(0.2, True)]

		for i in range(0, nlayer-1):
			in_nc = min(ndf*(2**(i)), self.feature_dim)
			out_nc = min(ndf*(2**(i+1)), self.feature_dim)
			model += [nn.Conv2d(in_nc, out_nc, kernel_size=self.k, stride=2, padding=self.pad, bias=self.use_bias), 
						norm(out_nc, affine=affine, track_running_stats=track_running_stats), nn.LeakyReLU(0.2, True)]

		self.model = nn.Sequential(*model)
		self.linear = nn.Linear(self.feature_dim, 1, bias=True)

	def forward(self, input_):
		feature = self.model(input_)

		N, C, H, W = feature.size()
		if self.pool == 'average':
			feature = F.avg_pool2d(feature, (H, W))
		elif self.pool == 'max':
			feature = F.max_pool2d(feature, (H, W))

		feature = feature.view(N, C)
		logits = self.linear(feature)

		return logits

if __name__ == '__main__':
	pass
	#



















