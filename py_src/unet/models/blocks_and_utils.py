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
from typing import List

class InitWeights_He(object):
	def __init__(self, neg_slope=1e-2):
		self.neg_slope = neg_slope

	def __call__(self, module):
		if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
			module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
			if module.bias is not None:
				module.bias = nn.init.constant_(module.bias, 0)

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, #BatchNorm2d
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, #BatchNorm2d
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers.
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


class ResidualBlock(nn.Module):
	expansion: int = 1
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride = 1) -> None:
		super(ResidualBlock, self).__init__()
		self.input_channels = in_channels
		self.output_channels = out_channels
		# downsample = out_channels != in_channels
		downsample = True
		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1

		self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False) if downsample else None
		norm_layer = nn.BatchNorm2d #BatchNorm2d
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding_size, bias=False)
		self.bn1 = norm_layer(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding_size, bias=False)
		self.bn2 = norm_layer(out_channels)
		self.stride = 1
	
	def forward(self, x: Tensor) -> Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out
				
class InceptionBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride = 1) -> None:
		super(InceptionBlock, self).__init__()
		self.input_channels = in_channels
		self.output_channels = out_channels
		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1
		conv_block = BasicConv2d
		norm_layer = nn.BatchNorm2d #BatchNorm2d
		self.relu = nn.ReLU(inplace=True)
		self.branch1x1 = conv_block(in_channels, out_channels, kernel_size=1)
		
		self.branch3x3_1 = conv_block(in_channels, out_channels, kernel_size=1)
		self.branch3x3_2 = conv_block(out_channels, out_channels, kernel_size=kernel_size, padding=padding_size)
		self.branch3x3_bn = norm_layer(out_channels)
		self.branch3x3_2a = conv_block(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding_size))
		self.branch3x3_2b = conv_block(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding_size, 0))
		self.branch3x3_bn2 = norm_layer(2*out_channels)

		# self.branch3x3dbl_1 = conv_block(in_channels, out_channels, kernel_size=1)
		# self.branch3x3dbl_3a = conv_block(out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, 1))
		# self.branch3x3dbl_3b = conv_block(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(1, 0))
		
		self.branch_pool = conv_block(in_channels, out_channels, kernel_size=1)
		self.dim_red = conv_block(out_channels * 4, out_channels, kernel_size=1 if stride == 1 else 3, stride=stride, padding=0 if stride == 1 else 1)
		self.bn = norm_layer(out_channels)

	def forward(self, x: Tensor) -> Tensor:
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)
		branch3x3 = self.branch3x3_bn(branch3x3)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)
		branch3x3 = self.branch3x3_bn2(branch3x3)

		# branch3x3dbl = self.branch3x3dbl_1(x)
		# branch3x3dbl = [
		# 	self.branch3x3dbl_3a(branch3x3dbl),
		# 	self.branch3x3dbl_3b(branch3x3dbl),
		# ]
		# branch3x3dbl = torch.cat(branch3x3dbl, 1)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		out = [branch1x1, branch3x3, branch_pool]  #branch3x3dbl,
		out = torch.cat(out, 1)
		out = self.dim_red(out)
		out = self.bn(out)
		out = self.relu(out)
		return out

class MobileBlock(_MobileBlock):
	def __init__(self, input_features: int, output_features: int, kernel_size: int = 3, stride = 1) -> None:
		cnf = InvertedResidualConfig(input_features, kernel_size, 2*max(input_features,output_features), output_features, True, "RE", stride, 1, 1.0)
		norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
		super().__init__(cnf, norm_layer)
		self.input_channels = input_features
		self.output_channels = output_features
		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1
		self.block = nn.Sequential(self.block, nn.Conv2d(output_features, output_features, kernel_size=kernel_size, padding=padding_size))

class DenseLayer(nn.Module):
	def __init__(self, input_features: int, output_features: int, kernel_size: int = 3, stride = 1) -> None:
		super(DenseLayer, self).__init__()
		self.input_channels = input_features
		self.output_channels = output_features
		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1
		self.norm1: nn.BatchNorm2d #BatchNorm2d
		self.add_module('norm1', nn.BatchNorm2d(input_features))
		self.relu1: nn.ReLU
		self.add_module('relu1', nn.ReLU(inplace=True))
		self.conv1: nn.Conv2d
		self.add_module('conv1', nn.Conv2d(input_features, output_features, kernel_size=(1,1), stride=1, bias=False))
		self.norm2: nn.BatchNorm2d # BatchNorm2d
		self.add_module('norm2', nn.BatchNorm2d(output_features))
		self.relu2: nn.ReLU
		self.add_module('relu2', nn.ReLU(inplace=True))
		self.conv2: nn.Conv2d
		self.add_module('conv2', nn.Conv2d(output_features, output_features, kernel_size=kernel_size, stride=stride, 
										padding=padding_size, bias=False))

	def bn_function(self, inputs: List[Tensor]) -> Tensor:
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
		return bottleneck_output

	def forward(self, input: Tensor) -> Tensor:
		if isinstance(input, Tensor):
			prev_features = [input]
		else:
			prev_features = input
		bottleneck_output = self.bn_function(prev_features)
		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		return new_features

class DenseBlock(nn.Module):
	def __init__(self, input_features: int, output_features: int, kernel_size: int = 3, stride = 1) -> None:
		super(DenseBlock, self).__init__()
		self.input_channels = input_features
		self.output_channels = output_features
		num_layers = 2
		conv_block = BasicConv2d
		# self.input_red = conv_block(input_features, output_features, kernel_size=1 if stride==1 else 3, stride=stride,  padding=0 if stride == 1 else 1)
		self.bn = nn.BatchNorm2d(output_features) #BatchNorm2d
		self.relu = nn.ReLU(inplace=True)
		self.dim_red = conv_block(num_layers*output_features + input_features, output_features, kernel_size=1 if stride==1 else 3, stride=stride,  padding=0 if stride == 1 else 1)
		self.items = []
		for i in range(num_layers):
			layer = DenseLayer(
				input_features + i * output_features,
				output_features,
				kernel_size=kernel_size)
			self.items.append(layer)
		self.items = nn.ModuleList(self.items)

	def forward(self, init_features: Tensor) -> Tensor:
		features = [init_features]
		for i, layer in enumerate(self.items):
			new_features = layer(features)
			features.append(new_features)
		features = torch.cat(features, 1)

		out = self.dim_red(features)
		out = self.bn(out)
		out = self.relu(out)
		

		return out

class VGGBlock(nn.Module):
	def __init__(self, input_feature_channels, output_feature_channels, num_convs = 2, kernel_size=3,
					conv_op=nn.Conv2d, conv_kwargs=None,
					norm_op=nn.BatchNorm2d, norm_op_kwargs=None, #BatchNorm2d
					dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
					nonlin=nn.LeakyReLU, nonlin_kwargs=None, stride=1, basic_block=ConvDropoutNormNonlin):

		self.input_channels = input_feature_channels
		self.output_channels = output_feature_channels

		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1

		if nonlin_kwargs is None:
			nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
		if dropout_op_kwargs is None:
			dropout_op_kwargs = {'p': 0.0, 'inplace': True} 
		if norm_op_kwargs is None:
			norm_op_kwargs = {'eps': 1e-5, 'affine': True} #, 'momentum': 0.1
		if conv_kwargs is None:
			conv_kwargs = {'kernel_size': kernel_size, 'stride': 1, 'padding': [padding_size, padding_size], 'dilation': 1, 'bias': True}

		self.nonlin_kwargs = nonlin_kwargs
		self.nonlin = nonlin
		self.dropout_op = dropout_op
		self.dropout_op_kwargs = dropout_op_kwargs
		self.norm_op_kwargs = norm_op_kwargs
		self.conv_kwargs = conv_kwargs
		self.conv_op = conv_op
		self.norm_op = norm_op

		if stride is not None:
			self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
			self.conv_kwargs_first_conv['stride'] = stride
		else:
			self.conv_kwargs_first_conv = conv_kwargs

		super(VGGBlock, self).__init__()
		self.blocks = nn.Sequential(
			*([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
							self.conv_kwargs_first_conv,
							self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
							self.nonlin, self.nonlin_kwargs)] +
				[basic_block(output_feature_channels, output_feature_channels, self.conv_op,
							self.conv_kwargs,
							self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
							self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

	def forward(self, x):
		return self.blocks(x)

class VGGBlock2(nn.Module):
	def __init__(self, input_feature_channels, output_feature_channels, num_convs = 2, kernel_size=3,
					conv_op=nn.Conv2d, conv_kwargs=None,
					norm_op=nn.BatchNorm2d, norm_op_kwargs=None, #BatchNorm2d
					dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
					nonlin=nn.LeakyReLU, nonlin_kwargs=None, stride=1, basic_block=ConvDropoutNormNonlin):

		self.input_channels = input_feature_channels
		self.output_channels = output_feature_channels

		padding_size = 2 if kernel_size == 5 else 3 if kernel_size == 7 else 1

		if nonlin_kwargs is None:
			nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
		if dropout_op_kwargs is None:
			dropout_op_kwargs = {'p': 0.0, 'inplace': True} 
		if norm_op_kwargs is None:
			norm_op_kwargs = {'eps': 1e-5, 'affine': True} #, 'momentum': 0.1
		if conv_kwargs is None:
			conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': [1,1], 'dilation': 1, 'bias': True}

		self.nonlin_kwargs = nonlin_kwargs
		self.nonlin = nonlin
		self.dropout_op = dropout_op
		self.dropout_op_kwargs = dropout_op_kwargs
		self.norm_op_kwargs = norm_op_kwargs
		self.conv_kwargs = conv_kwargs
		self.conv_op = conv_op
		self.norm_op = norm_op

		if stride is not None:
			self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
			self.conv_kwargs_first_conv['stride'] = stride
			self.conv_kwargs_first_conv['kernel_size'] = kernel_size
			self.conv_kwargs_first_conv['padding'] = [padding_size, padding_size]
		else:
			self.conv_kwargs_first_conv = conv_kwargs

		super(VGGBlock2, self).__init__()
		self.blocks = nn.Sequential(
			*([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
							self.conv_kwargs_first_conv,
							self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
							self.nonlin, self.nonlin_kwargs)] +
				[basic_block(output_feature_channels, output_feature_channels, self.conv_op,
							self.conv_kwargs,
							self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
							self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

	def forward(self, x):
		return self.blocks(x)


if __name__ == '__main__':
	device = "cpu"
	input1 = torch.rand(1, 16, 128, 128).to(device)
	model = InceptionBlock(16, 32)
	model2 = ResidualBlock(32, 64)
	model2a = InceptionBlock(64, 64)
	model2b = ResidualBlock(64, 64)
	model2c = VGGBlock(64, 64)
	model2d = MobileBlock(64, 64)
	model2e = DenseBlock(64, 64)
	model3 = VGGBlock(64, 128)
	model4 = MobileBlock(128, 256)
	model5 = DenseBlock(256, 512)
	model6a = InceptionBlock(512, 256)
	model6b = ResidualBlock(256, 128)
	model6c = VGGBlock(128, 64)
	model6d = MobileBlock(64, 32)
	model6e = DenseBlock(32, 16)
	output = model(input1)
	print(output.shape) 
	output = model2e(model2d(model2c(model2b(model2a(model2(output))))))
	print(output.shape) 
	output = model3(output)
	print(output.shape) 
	output = model4(output)
	print(output.shape)
	output = model5(output)
	print(output.shape) 
	output = model6e(model6d(model6c(model6b(model6a(output)))))
	print(output.shape)