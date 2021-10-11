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

class NAS_WNet(nn.Module):

	def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
					feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
					norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
					dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
					nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False, 
					weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,conv_kernel_sizes=None,
					upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
					max_num_features=None, basic_block=ConvDropoutNormNonlin, depth=None, skip_connects=None):
		super(NAS_WNet, self).__init__()

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
		self.depth = depth

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
		self.td = []
		self.tu = []
		self.seg_outputs = []
			
		print("UNET-py: Network depth per layer:", depth)
		print("UNET-py: Network conv sizes", conv_kernel_sizes)
		print("UNET-py: Skip connects", skip_connects)

		# Calculate scaling, input and output features and skipconnects 
		output_features = base_num_features
		input_features = input_channels

		self.connections = {0: {"depth": depth[0], "a": output_features, "b": output_features, "skip_from": [], "skip_to": []}}
		feats = output_features
		for i in range(1, len(depth)):
			same_level = []
			for j in range(max(0,i-2), -1, -1):
				if depth[j] == depth[i]:
					same_level.append(j)
			skip_from = []
			skip_multiplier = 1
			if skip_connects[i] == 1:
				skip_from = same_level[:1]
			elif skip_connects[i] == 2:
				skip_from = same_level[1:2]
			elif skip_connects[i] == 3:
				skip_from = same_level[:2]
			skip_multiplier += len(skip_from)
			
			if depth[i] - depth[i - 1] == 1:
				feats = min(2 * feats, self.max_num_features)
			elif depth[i] - depth[i - 1] == -1:
				feats = feats // 2
			self.connections[i] = {"depth": depth[i], "a": feats, "b": feats*skip_multiplier, "skip_from": skip_from, "skip_to": []} 
			for source in skip_from:
				self.connections[source]["skip_to"].append(i)
			
		scaling = "down"
		# Add downsampling at start
		self.network = {0: {"input": input_features, "output": output_features, "scaling": scaling}}
		for i in self.connections:
			if i != 0:
				input_features = self.connections[i-1]["b"]
				output_features = self.connections[i]["a"]
				if self.connections[i]["a"] // self.connections[i-1]["a"] == 1:
					scaling = "non"
				elif self.connections[i]["a"] // self.connections[i-1]["a"] > 1:
					scaling = "down"
				else:
					scaling = "up"
					input_features = input_features // 2
				self.network[i] = {"input": input_features, "output": output_features, "scaling": scaling}

		self.network[len(self.connections)] = {"input": self.connections[len(self.connections)-1]["b"], "output": base_num_features, "scaling": "non"}

		for l in range(len(self.network)):
			print("UNET-py: block", l, self.network[l])
			if l in self.connections:
				print("UNET-py: connection", l, self.connections[l])

		# add convolutions
		# initial downscale - always added
		first_stride = None
		self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[0]
		self.conv_kwargs['padding'] = self.conv_pad_sizes[0]	

		for d in range(len(self.network)):
			layer = self.network[d]
			if layer["scaling"] == "down":
				# downsampling block
				input_features, output_features = layer["input"], layer["output"]
				if self.convolutional_pooling and d != 0:
					first_stride = pool_op_kernel_sizes[d]
				self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d-1] if d>0 else [3,3]
				self.conv_kwargs['padding'] = self.conv_pad_sizes[d-1] if d>0 else [1,1]
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
					self.tu.append(transpconv(self.connections[d-1]["b"], layer["input"], pool_op_kernel_sizes[0],
											pool_op_kernel_sizes[0], bias=False))
				self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d-1]
				self.conv_kwargs['padding'] = self.conv_pad_sizes[d-1]
				self.conv_blocks.append(nn.Sequential(
					StackedConvLayers(input_features, output_features, num_conv_per_stage - 1,
									self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
									self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
					StackedConvLayers(output_features, output_features, 1, self.conv_op, self.conv_kwargs,
									self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
									self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
		

		for l in range(len(self.network)):
			if self.network[l]["scaling"] == "up":
				ds = l
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
					self.upscale_logits_ops) 

		if self.weightInitializer is not None:
			self.apply(self.weightInitializer)

	def forward(self, x):
		tu_count = 0
		skips = {}
		seg_outputs = []
		for l in range(len(self.network)-1):
			layer = self.network[l]
			next_layer = self.network[l+1]
			x = self.conv_blocks[l](x)
			skips[l] = x
			# print("tensor: ", x.size(), "; layer:", l, "; depth: ", self.depth[l])
			if layer["scaling"] == "up":
				seg_outputs.append(self.seg_outputs[tu_count](x))
				tu_count += 1
			connections = self.connections[l]
			for source in connections["skip_from"]:
				# print("skip: ", skips[source].size(), "; layer:", l, "; depth: ", self.depth[l])
				x = torch.cat((x, skips[source]), dim=1)
			if next_layer["scaling"] == "up":
				x = self.tu[tu_count](x)
			# print("tensor: ", x.size(), "; layer:", l, "; depth: ", self.depth[l])
		x = self.conv_blocks[-1](x)
		# print("tensor: ", x.size(), "; layer:", l, "; depth: ", self.depth[l])
		seg_outputs.append(self.seg_outputs[-1](x))

		if self.do_ds:
			return tuple([seg_outputs[-1]] + [i(j) for i, j in
						zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
		else:
			return seg_outputs[-1]

class NAS_multipath_WNet(nn.Module):

	def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
					feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
					norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
					dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
					nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False, 
					weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,conv_kernel_sizes=None,
					upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
					max_num_features=None, basic_block=ConvDropoutNormNonlin, depth=None, skip_connects=None,
					n_layers_path=10):
		super(NAS_multipath_WNet, self).__init__()

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
		self.depth = depth
		self.num_paths = len(self.depth) // n_layers_path

		if conv_op == nn.Conv2d:
			upsample_mode = 'bilinear'
			pool_op = nn.MaxPool2d
			transpconv = nn.ConvTranspose2d
			if pool_op_kernel_sizes is None:
				pool_op_kernel_sizes = [(2, 2)] * num_pool
			if conv_kernel_sizes is None:
				conv_kernel_sizes = [(3, 3)] * (num_pool * self.num_paths)
		elif conv_op == nn.Conv3d:
			upsample_mode = 'trilinear'
			pool_op = nn.MaxPool3d
			transpconv = nn.ConvTranspose3d
			if pool_op_kernel_sizes is None:
				pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
			if conv_kernel_sizes is None:
				conv_kernel_sizes = [(3, 3, 3)] * (num_pool * self.num_paths)
		if skip_connects is None:
			skip_connects = [1] * num_pool
		self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
		self.pool_op_kernel_sizes = pool_op_kernel_sizes
		self.conv_kernel_sizes = conv_kernel_sizes
		self.skip_connects = skip_connects

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
		self.connections = [{} for _ in range(self.num_paths)]
		self.network = [{} for _ in range(self.num_paths)]
		self.conv_blocks = [[] for _ in range(self.num_paths)]
		self.td = [[] for _ in range(self.num_paths)]
		self.tu = [[] for _ in range(self.num_paths)]
		self.seg_outputs = [[] for _ in range(self.num_paths)]
			
		print("UNET-py: Network depth per layer:", depth)
		print("UNET-py: Network conv sizes", conv_kernel_sizes)
		print("UNET-py: Skip connects", skip_connects)

		for p in range(self.num_paths):
			# Calculate scaling, input and output features and skipconnects 
			output_features = base_num_features
			input_features = input_channels
			depth = self.depth[p * n_layers_path:(p+1) * n_layers_path]
			skip_connects = self.skip_connects[p * n_layers_path:(p+1) * n_layers_path]
			pool_op_kernel_sizes = self.pool_op_kernel_sizes[max(0, p * n_layers_path - 1) : (p+1) * n_layers_path - 1]
			conv_kernel_sizes = self.conv_kernel_sizes[p * n_layers_path:(p+1) * n_layers_path]
			conv_pad_sizes = self.conv_pad_sizes[p * n_layers_path:(p+1) * n_layers_path]

			self.connections[p] = {0: {"depth": depth[0], "a": output_features, "b": output_features, "skip_from": [], "skip_to": []}}
			feats = output_features
			for i in range(1, len(depth)):
				same_level = []
				for j in range(max(0,i-2), -1, -1):
					if depth[j] == depth[i]:
						same_level.append(j)
				skip_from = []
				skip_multiplier = 1
				if skip_connects[i] == 1:
					skip_from = same_level[:1]
				elif skip_connects[i] == 2:
					skip_from = same_level[1:2]
				elif skip_connects[i] == 3:
					skip_from = same_level[:2]
				skip_multiplier += len(skip_from)
				
				if depth[i] - depth[i - 1] == 1:
					feats = min(2 * feats, self.max_num_features)
				elif depth[i] - depth[i - 1] == -1:
					feats = feats // 2
				self.connections[p][i] = {"depth": depth[i], "a": feats, "b": feats*skip_multiplier, "skip_from": skip_from, "skip_to": []} 
				for source in skip_from:
					self.connections[p][source]["skip_to"].append(i)
				
			scaling = "down"
			# Add downsampling at start
			self.network[p] = {0: {"input": input_features, "output": output_features, "scaling": scaling}}
			for i in self.connections[p]:
				if i != 0:
					input_features = self.connections[p][i-1]["b"]
					output_features = self.connections[p][i]["a"]
					if self.connections[p][i]["a"] // self.connections[p][i-1]["a"] == 1:
						scaling = "non"
					elif self.connections[p][i]["a"] // self.connections[p][i-1]["a"] > 1:
						scaling = "down"
					else:
						scaling = "up"
						input_features = input_features // 2
					self.network[p][i] = {"input": input_features, "output": output_features, "scaling": scaling}

			if p == self.num_paths - 1:
				size_after_path_concat = 0 
				for p2 in range(self.num_paths):
					size_after_path_concat += self.connections[p2][len(self.connections[p2])-1]["b"]
				self.network[p][len(self.connections[p])] = {"input": size_after_path_concat, "output": base_num_features, "scaling": "non"}

			for l in range(len(self.network[p])):
				print("UNET-py: block", l, self.network[p][l])
				if l in self.connections[p]:
					print("UNET-py: connection", l, self.connections[p][l])

			first_stride = None
			for d in range(len(self.network[p])):
				layer = self.network[p][d]
				if layer["scaling"] == "down":
					# downsampling block
					input_features, output_features = layer["input"], layer["output"]
					if self.convolutional_pooling and d != 0:
						first_stride = pool_op_kernel_sizes[d]
					self.conv_kwargs['kernel_size'] = conv_kernel_sizes[d-1] if d>0 else [3,3]
					self.conv_kwargs['padding'] = conv_pad_sizes[d-1] if d>0 else [1,1]
					self.conv_blocks[p].append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
																	self.conv_op, self.conv_kwargs, self.norm_op,
																	self.norm_op_kwargs, self.dropout_op,
																	self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
																	first_stride, basic_block=basic_block))
					if not self.convolutional_pooling:
						self.td[p].append(pool_op(pool_op_kernel_sizes[d]))

				elif layer["scaling"] == "non":
					# Nonscaling block
					input_features, output_features = layer["input"], layer["output"]
					first_stride = None
					self.conv_kwargs['kernel_size'] = conv_kernel_sizes[d-1]
					self.conv_kwargs['padding'] = conv_pad_sizes[d-1]
					self.conv_kwargs['stride'] = 1
					self.conv_blocks[p].append(nn.Sequential(
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
						self.tu[p].append(Upsample(scale_factor=pool_op_kernel_sizes[0], mode=upsample_mode))
					else:
						self.tu[p].append(transpconv(self.connections[p][d-1]["b"], layer["input"], pool_op_kernel_sizes[0],
												pool_op_kernel_sizes[0], bias=False))
					self.conv_kwargs['kernel_size'] = conv_kernel_sizes[d-1]
					self.conv_kwargs['padding'] = conv_pad_sizes[d-1]
					# print("UNET-py: UPSCALE input: ", input_features, "; output: ", output_features)
					self.conv_blocks[p].append(nn.Sequential(
						StackedConvLayers(input_features, output_features, num_conv_per_stage - 1,
										self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
										self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
						StackedConvLayers(output_features, output_features, 1, self.conv_op, self.conv_kwargs,
										self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
										self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
			

		self.seg_outputs[p].append(conv_op(self.conv_blocks[p][-1][-1].output_channels, num_classes,
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
		for p in range(self.num_paths):
			self.conv_blocks[p] = nn.ModuleList(self.conv_blocks[p])
			self.td[p] = nn.ModuleList(self.td[p])
			self.tu[p] = nn.ModuleList(self.tu[p])
			self.seg_outputs[p] = nn.ModuleList(self.seg_outputs[p])
			if self.upscale_logits:
				self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops) 

		if self.weightInitializer is not None:
			self.apply(self.weightInitializer)

	def forward(self, x):
		skips = [{} for _ in range(self.num_paths)]
		seg_outputs = []
		y = [x for _ in range(self.num_paths)]
		for p in range(self.num_paths):
			tu_count = 0
			for l in range(len(self.network[p])-1):
				layer = self.network[p][l]
				next_layer = self.network[p][l+1]
				y[p] = self.conv_blocks[p][l](y[p])
				skips[p][l] = y[p]
				# print("tensor: ", y[p].size(), "; layer:", l, "; depth: ", self.depth[l])
				if layer["scaling"] == "up":
					# seg_outputs.append(self.seg_outputs[p][tu_count](x))
					tu_count += 1
				connections = self.connections[p][l]
				for source in connections["skip_from"]:
					# print("skip: ", skips[p][source].size(), "; layer:", l, "; depth: ", self.depth[l])
					y[p] = torch.cat((y[p], skips[p][source]), dim=1)
				if next_layer["scaling"] == "up":
					y[p] = self.tu[p][tu_count](y[p])
				# print("tensor: ", x.size(), "; layer:", l, "; depth: ", self.depth[l])
		print("tensor: ", y[0].size(), y[1].size(), "; layer:", l, "; depth: ", self.depth[l])
		y = torch.cat((y[0], y[1]), dim=1)
		x = self.conv_blocks[p][-1](y)
		# print("tensor: ", x.size(), "; layer:", l, "; depth: ", self.depth[l])
		seg_outputs.append(self.seg_outputs[p][-1](x))

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
	model = NAS_multipath_WNet(1, 32, 1, 18,
                        2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                        dropout_op_kwargs,
                        net_nonlin, net_nonlin_kwargs, False, False, InitWeights_He(1e-2),
                        None, None, False, True, True, 
						depth=[0,1,2,3,3,3,3,2,1,0, 0,1,2,3,2,1,2,1,1,0], #, 0, 1, 2, 1, 2, 1, 1, 2, 1, 0], 
						skip_connects=[0,0,0,0,0,1,2,1,1,1,0,0,0,0,0,1,2,1,1,1]).to(device) #,0,0,0,0,0,2,2,1,1,0]).to(device)
	output = model(input1)
	print(output.shape) 