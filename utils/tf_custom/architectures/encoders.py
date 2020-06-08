"""This file contains the encoders that can be used in a VAE 
"""
from utils.tf_custom.architectures import base
import tensorflow as tf
from utils.tf_custom.architectures import architecture_params as ap

class _GaussianEncoder(base.ConvolutionalNeuralNetwork):
	def __init__(self, layer_params, num_latents, shape_input, activations=None, **kwargs):
		"""Base clase for a gaussian encoder
		
		Args:
			layer_params (list): layer size specs
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
			activations (dict): this is a dictionary of activations
		"""
		self.shape_input = shape_input
		self.num_latents = num_latents
		if activations is None:
			activations = ap.default_encoder_activations
		layer_params = layer_params + [[self.num_latents*2]]

		self._configuration_parameters = dict(
					layer_params=layer_params, 
					shape_input=self.shape_input,
					num_latents=self.num_latents
					)

		super().__init__(*layer_params, activation=activations, shape_input=self.shape_input)
	
	def call(self,inputs):
		assert list(inputs.shape[1:]) == self.shape_input, "%s, %s"%(inputs.shape[1:], self.shape_input)
		out = super().call(inputs)
		mean = out[:,:self.num_latents]
		logvar = out[:,self.num_latents:]
		sample = tf.exp(0.5*logvar)*tf.random.normal(
			tf.shape(logvar))+mean
		return sample, mean, logvar

	def get_config(self):
		return {**base.convert_config(self._configuration_parameters), 
				"activations":base.convert_config(self._total_activations)} # activations are separately added 

class GaussianEncoder64(_GaussianEncoder):
	def __init__(self, num_latents=10, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		self.shape_input = [64,64,3]
		self.layer_params = ap.simple64_layer_parameters[:]
		if "num_channels" in kwargs:
			self.shape_input[-1] = kwargs["num_channels"]
		super().__init__(self.layer_params, num_latents, self.shape_input, activations, **kwargs)

class GaussianEncoder256(_GaussianEncoder):
	def __init__(self, num_latents=30, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 512x512x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		self.shape_input = [256,256,3]
		self.layer_params = ap.hq256_layer_parameters[:]
		if "num_channels" in kwargs:
			self.shape_input[-1] = kwargs["num_channels"]
		super().__init__(self.layer_params, num_latents, self.shape_input, activations, **kwargs)

class GaussianEncoder512(_GaussianEncoder):
	def __init__(self, num_latents=1024, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 512x512x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		self.shape_input = [512,512,3]
		self.layer_params = ap.hq512_layer_parameters[:]
		if "num_channels" in kwargs:
			self.shape_input[-1] = kwargs["num_channels"]
		super().__init__(self.layer_params, num_latents, self.shape_input, activations, **kwargs)


def main():
	import numpy as np
	inputs = np.random.randint(0,255,size=[8,64,64,3], dtype=np.uint8).astype(np.float32)
	encoder = GaussianEncoder64()
	print(encoder(inputs)[0])
	print(encoder.shape_before_flatten)

if __name__ == '__main__':
	main()