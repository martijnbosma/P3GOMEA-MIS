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
from unet.models.blocks_and_utils import *

class NAS_UNet(nn.Module):

	def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
					feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
					norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
					dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
					nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False, 
					weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,conv_kernel_sizes=None,
					upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
					max_num_features=None, basic_block=ConvDropoutNormNonlin, depth=None, skip_connects=None):
		super(NAS_UNet, self).__init__()

		# set params 
		self.convolutional_upsampling = convolutional_upsampling
		self.convolutional_pooling = convolutional_pooling
		self.upscale_logits = upscale_logits
		if nonlin_kwargs is None:
			nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
		if dropout_op_kwargs is None:
			dropout_op_kwargs = {'p': 0.5, 'inplace': True}
		if norm_op_kwargs is None:
			norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
		self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
		self.nonlin = nonlin
		self.nonlin_kwargs = nonlin_kwargs
		self.dropout_op_kwargs = dropout_op_kwargs
		self.norm_op_kwargs = norm_op_kwargs
		self.weightInitializer = weightInitializer
		self.conv_op = conv_op
		self.norm_op = norm_op
		self.dropout_op = dropout_op
		self.num_classes = num_classes
		self._deep_supervision = deep_supervision
		self.do_ds = deep_supervision

		if conv_op == nn.Conv2d:
			upsample_mode = 'bilinear'
			pool_op = nn.MaxPool2d
			transpconv = nn.ConvTranspose2d
			if pool_op_kernel_sizes is None:
				pool_op_kernel_sizes = [(2, 2)] * num_pool
			if conv_kernel_sizes is None:
				conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
		elif conv_op == nn.Conv3d:
			upsample_mode = 'trilinear'
			pool_op = nn.MaxPool3d
			transpconv = nn.ConvTranspose3d
			if pool_op_kernel_sizes is None:
				pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
			if conv_kernel_sizes is None:
				conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
		if skip_connects is None:
			skip_connects = [1] * num_pool
		self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
		self.pool_op_kernel_sizes = pool_op_kernel_sizes
		self.conv_kernel_sizes = conv_kernel_sizes

		self.conv_pad_sizes = []
		for krnl in self.conv_kernel_sizes:
			self.conv_pad_sizes.append([1 if i == 3 else 2 if i == 5 else 3 if i == 7 else 0 for i in krnl])

		if max_num_features is None:
			if self.conv_op == nn.Conv3d:
				self.max_num_features = 300
			else:
				self.max_num_features = 512
		else:
			self.max_num_features = max_num_features

		# Instantiate variables
		self.conv_blocks = []
		self.downscaling = []
		self.upscaling = []
		self.nonscaling = []
		self.td = []
		self.tu = []
		self.seg_outputs = []
		self.down_up = None
		self.without_bottleneck = False


		# Calculate scaling, input and output features and skipconnects 
		output_features = base_num_features
		input_features = input_channels
		scaling = "down"
		# Add downsampling at start
		self.network = {0: {"input": input_features, "output": output_features, 
				"skip_from": [], "skip_to": [], "scaling": scaling}}
		self.downscaling.append(0)
		print("UNET-py: Network depth per layer:", depth)

		for i in range(1, len(depth)):
			same_level = []
			for j in range(i-2, -1, -1):
				if depth[j] == depth[i]:
					same_level.append(j)
			skip_from = []
			skip_to = []
			if skip_connects[i] == 1:
				skip_from = same_level[:1]
			elif skip_connects[i] == 2:
				skip_from = same_level[:2]
			skip_multiplier = 1 + skip_connects[i]
			if depth[i] - depth[i - 1] == 1:
				input_features = output_features
				output_features = int(np.round(input_features * feat_map_mul_on_downscale))
				output_features = min(output_features, self.max_num_features)
				input_features = input_features * skip_multiplier
				self.downscaling.append(i)
				scaling = "down"
			elif depth[i] - depth[i - 1] == 0:
				input_features = output_features * skip_multiplier
				self.nonscaling.append(i)
				scaling = "non"
			else:
				input_features = output_features
				output_features = int(np.round(input_features / feat_map_mul_on_downscale))
				input_features = output_features * skip_multiplier
				self.upscaling.append(i)
				scaling = "up"
			self.network[i] = {"input": input_features, "output": output_features, 
					"skip_from": skip_from, "skip_to": skip_to, "scaling": scaling}
			for source in skip_from:
				self.network[source]["skip_to"].append(i)

		for l in range(len(self.network)):
			print("UNET-py:", l, self.network[l])

		# add convolutions
		# initial downscale - always added
		first_stride = None
		self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[0]
		self.conv_kwargs['padding'] = self.conv_pad_sizes[0]	

		for d in range(0, len(self.network)):
			layer = self.network[d]
			if layer["scaling"] == "down":
				# downsampling block
				input_features, output_features = layer["input"], layer["output"]
				if self.convolutional_pooling and d != 0:
					first_stride = pool_op_kernel_sizes[d]
				self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
				self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
				# print("UNET-py: DOWNSCALE input: ", input_features, "; output: ", output_features)
				self.conv_blocks.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
																self.conv_op, self.conv_kwargs, self.norm_op,
																self.norm_op_kwargs, self.dropout_op,
																self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
																first_stride, basic_block=basic_block))
				if not self.convolutional_pooling:
					self.td.append(pool_op(pool_op_kernel_sizes[d]))

			elif layer["scaling"] == "non":
				# Nonscaling block
				input_features, output_features = layer["input"], layer["output"]
				first_stride = None
				self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d-1]
				self.conv_kwargs['padding'] = self.conv_pad_sizes[d-1]
				self.conv_kwargs['stride'] = 1
				# print("UNET-py: NONSCALE input: ", input_features, "; output: ", output_features)
				self.conv_blocks.append(nn.Sequential(
					StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
									self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
									self.nonlin_kwargs, first_stride, basic_block=basic_block),
					StackedConvLayers(output_features, output_features, 1, self.conv_op, self.conv_kwargs,
									self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
									self.nonlin_kwargs, basic_block=basic_block)))

			else:
				# Upscaling block
				input_features, output_features = layer["input"], layer["output"]
				if not self.convolutional_upsampling:
					self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[0], mode=upsample_mode))
				else:
					self.tu.append(transpconv(self.network[d-1]["output"], layer["output"], pool_op_kernel_sizes[0],
											pool_op_kernel_sizes[0], bias=False))
				self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d-1]
				self.conv_kwargs['padding'] = self.conv_pad_sizes[d-1]
				# print("UNET-py: UPSCALE input: ", input_features, "; output: ", output_features)
				self.conv_blocks.append(nn.Sequential(
					StackedConvLayers(input_features, output_features, num_conv_per_stage - 1,
									self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
									self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
					StackedConvLayers(output_features, output_features, 1, self.conv_op, self.conv_kwargs,
									self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
									self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
		

		for ds in self.upscaling:
			if ds < len(self.network):
				self.seg_outputs.append(conv_op(self.conv_blocks[ds][-1].output_channels, num_classes,
												1, 1, 0, 1, 1, True))
		self.seg_outputs.append(conv_op(self.conv_blocks[-1][-1].output_channels, num_classes,
												1, 1, 0, 1, 1, True))

		self.upscale_logits_ops = []
		cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
		for usl in range(max(depth) - 1):
			if self.upscale_logits:
				self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
														mode=upsample_mode))
			else:
				self.upscale_logits_ops.append(lambda x: x)

		# register all modules properly
		self.conv_blocks = nn.ModuleList(self.conv_blocks)
		self.td = nn.ModuleList(self.td)
		self.tu = nn.ModuleList(self.tu)
		self.seg_outputs = nn.ModuleList(self.seg_outputs)
		if self.upscale_logits:
			self.upscale_logits_ops = nn.ModuleList(
				self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

		if self.weightInitializer is not None:
			self.apply(self.weightInitializer)

		# print("UNET-py: Downscaling: ", self.downscaling)
		# print("UNET-py: Upscaling: ", self.upscaling)
		# print("UNET-py: Nonscaling: ", self.nonscaling)
		# print("UNET-py: Seg outputs: ", self.seg_outputs)
		# print("UNET-py: TU: ", self.tu)
		# for ds in self.upscaling:
		# 	print("UNET-py: Output size: ", self.conv_blocks[ds][-1].output_channels)

	def forward(self, x):
		tu_count = 0
		skips = {}
		seg_outputs = []
		for l in range(len(self.network)):
			# print("x input: ", x.size())
			layer = self.network[l]
			if layer["scaling"] == "up":
				x = self.tu[tu_count](x)
				# print("x after tu: ", x.size())
			for source in layer["skip_from"]:
				x = torch.cat((x, skips[source]), dim=1)
				# print("x after concat: ", x.size())
			x = self.conv_blocks[l](x)
			if layer["scaling"] == "up":
				seg_outputs.append(self.seg_outputs[tu_count](x))
				tu_count += 1
			if len(layer["skip_to"]) > 0:
				skips[l] = x
			# print("x output: ", x.size())
		if len(seg_outputs) == 0:
			seg_outputs.append(self.seg_outputs[tu_count](x))

		if self.do_ds:
			return tuple([seg_outputs[-1]] + [i(j) for i, j in
						zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
		else:
			return seg_outputs[-1]


if __name__ == '__main__':
	device = "cpu"
	input1 = torch.rand(1, 1, 256, 256).to(device)
	conv_op = nn.Conv2d
	dropout_op = nn.Dropout2d
	norm_op = nn.InstanceNorm2d
	norm_op_kwargs = {'eps': 1e-5, 'affine': True}
	dropout_op_kwargs = {'p': 0, 'inplace': True}
	net_nonlin = nn.LeakyReLU
	net_nonlin_kwargs = {'inplace': True}
	model = NAS_UNet(1, 32, 1, 9,
                        2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs,
                        net_nonlin, net_nonlin_kwargs, False, False, InitWeights_He(1e-2),
                        None, None, False, True, True, 
						depth=[0,1,2,3,3,3,3,2,1,0], #, 0, 1, 2, 1, 2, 1, 1, 2, 1, 0], 
						skip_connects=[0,0,0,0,0,1,2,1,1,1]).to(device) #,0,0,0,0,0,2,2,1,1,0]).to(device)
	output = model(input1)
	print(output.shape) 