"""This file contains the decoders that can be used in a VAE 
"""
from utils.tf_custom.architectures import base
import tensorflow as tf
from utils.tf_custom.architectures import architecture_params as ap

class _Decoder(base.DeconvolutionalNeuralNetwork):
	def __init__(self, layer_params, shape_input, activations=None, shape_before_flatten=None):
		"""Base class for decoders
		
		Args:
			layer_params (list): layer size specs
			shape_input (list): the shape of the input image (not including batch size)
			activations (dict): this is a dictionary of activations
		    shape_before_flatten (None, list): shape of activations before flattening.
		"""
		self.shape_input = shape_input
		self.shape_before_flatten = shape_before_flatten
		if activations is None:
			activations = ap.default_activations
		super().__init__(*layer_params, activation=activations, shape_before_flatten=shape_before_flatten)
	
	def call(self, latent_elements):
		out = super().call(latent_elements)
		assert list(out.shape[1:]) == self.shape_input
		return out

class Decoder64(_Decoder):
	shape_input = [64,64,3]
	layer_params = ap.simple64_layer_parameters[::-1]
	shape_before_flatten = ap.simple64_shape_before_flatten
	def __init__(self, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.layer_params[-1][0] = self.shape_input[-1] # set num channels
		super().__init__(self.layer_params, self.shape_input, activations, self.shape_before_flatten)


class Decoder512(_Decoder):
	shape_input = [512,512,3]
	layer_params = ap.hq512_layer_parameters[::-1]
	shape_before_flatten = ap.hq512_shape_before_flatten
	def __init__(self, activations=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.layer_params[-1][0] = self.shape_input[-1] # set num channels
		super().__init__(self.layer_params, self.shape_input, activations, self.shape_before_flatten)


def main():
	import numpy as np
	inputs = np.random.randint(0,255,size=[8,10], dtype=np.uint8).astype(np.float32)
	encoder64 = Decoder64()
	encoder64(inputs)

	inputs = np.random.randint(0,255,size=[8,1024], dtype=np.uint8).astype(np.float32)
	encoder512 = Decoder512()
	encoder512(inputs)


if __name__ == '__main__':
	main()