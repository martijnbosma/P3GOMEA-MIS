import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor
from copy import deepcopy
from torchvision.models.mobilenetv3 import InvertedResidual as _MobileBlock
from torchvision.models.mobilenetv3 import InvertedResidualConfig
from torchvision.models.inception import BasicConv2d
from functools import partial

def weight_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)
	if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1.0)
		m.bias.data.fill_(0.0)
	if isinstance(m, nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)

def conv_block(in_ch=1, out_ch=1, threeD=True, batchnorm=False):
	if batchnorm:
		if threeD:
			layer = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.BatchNorm3d(out_ch),
									nn.LeakyReLU())
		else:
			layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
									nn.BatchNorm2d(out_ch),
									nn.LeakyReLU())
	else:
		if threeD:
			layer = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1), nn.LeakyReLU())
		else:
			layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.LeakyReLU())			
	return layer


def deconv_block(in_ch=1, out_ch=1, scale_factor=2, threeD=True, batchnorm=False):
	if batchnorm:
		if threeD:
			layer = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.BatchNorm3d(out_ch),
									nn.LeakyReLU())
		else:
			layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.BatchNorm2d(out_ch),
									nn.LeakyReLU())
	else:
		if threeD:
			layer = nn.Sequential(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.LeakyReLU())
		else:
			layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor),
									nn.LeakyReLU())
	return layer


def Unet_DoubleConvBlock(in_ch=1, out_ch=1, threeD=True, batchnorm=False):
	layer = nn.Sequential(conv_block(in_ch=in_ch, out_ch=out_ch, threeD=threeD, batchnorm=batchnorm),
							conv_block(in_ch=out_ch, out_ch=out_ch, threeD=threeD, batchnorm=batchnorm)
							)
	return layer

class UNet(nn.Module):
	"""
	Implementation of U-Net
	"""
	def __init__(self, depth=4, width=32, growth_rate=2, in_channels=1, out_channels=1, threeD=False, batchnorm=False):
		super(UNet, self).__init__()
		self.depth = depth
		self.out_channels = [width*(growth_rate**i) for i in range(self.depth+1)]

		# Downsampling Path Layers
		self.downblocks = nn.ModuleList()
		current_in_channels = in_channels
		for i in range(self.depth+1):
			self.downblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[i], threeD=threeD, batchnorm=batchnorm))
			current_in_channels = self.out_channels[i]

		self.feature_channels = current_in_channels + self.out_channels[i-1]
		# Upsampling Path Layers
		self.deconvblocks = nn.ModuleList()
		self.upblocks = nn.ModuleList()
		for i in range(self.depth):
			self.deconvblocks.append(deconv_block(current_in_channels, self.out_channels[-2 - i], threeD=threeD, batchnorm=batchnorm))
			self.upblocks.append(Unet_DoubleConvBlock(current_in_channels, self.out_channels[-2 - i], threeD=threeD, batchnorm=batchnorm))
			current_in_channels = self.out_channels[-2 - i]

		if threeD:
			self.last_layer = nn.Conv3d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool3d(2)
		else:
			self.last_layer = nn.Conv2d(current_in_channels, out_channels, kernel_size=1)
			self.downsample = nn.MaxPool2d(2)			

		# Initialization
		self.apply(weight_init)


	def forward(self, x):
		# Downsampling Path
		out = x
		down_features_list = list()
		for i in range(self.depth):
			out = self.downblocks[i](out)
			down_features_list.append(out)
			out = self.downsample(out)

		# bottleneck
		out = self.downblocks[-1](out)
		features = [down_features_list[-1], out]

		# Upsampling Path
		for i in range(self.depth):
			out = self.deconvblocks[i](out)
			down_features = down_features_list[-1 - i]
			
			# pad slice and image dimensions if necessary
			down_shape = torch.tensor(down_features.shape)
			out_shape = torch.tensor(out.shape)
			shape_diff = down_shape - out_shape
			pad_list = [padding for diff in reversed(shape_diff.numpy()) for padding in [diff,0]]
			if max(pad_list) == 1:
				out = F.pad(out, pad_list)

			out = torch.cat([down_features, out], dim=1)
			out = self.upblocks[i](out)

		out = self.last_layer(out)	

		return out


if __name__ == '__main__':
	device = "cpu"
	input1 = torch.rand(1, 1, 256, 256).to(device)
	model = UNet(5, 32, 2, 1, 1, False, True).to(device)
	output = model(input1)
	print(output.shape) 
