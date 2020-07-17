import copy
import tensorflow as tf
from . import architecture_params as ap
from . import base
from . import network

class Decoder(network.DeconvolutionalNeuralNetwork):
	def __init__(self, layer_params, num_latents, shape_image, activation, **kwargs):
		"""Base class for decoders
		
		Args:
			layer_params (list): layer size specs
			shape_image (list): the shape of the input image (not including batch size)
			activation (dict): this is a dictionary of activation
		    shape_before_flatten (None, list): shape of activation before flattening.
		"""
		# add additional attributes
		self.shape_image = shape_image
		self.num_latents = num_latents

		super().__init__(*layer_params, activation=activation, shape_input=[num_latents], **kwargs)
		
		self._config_param = dict(
					layer_params=layer_params, 
					shape_input=num_latents,
					shape_image=shape_image,
					activation=self.activation
					)

	def call(self, inputs):
		assert list(inputs.shape[1:]) == list(self.shape_input), "%s, %s"%(list(inputs.shape[1:]), self.shape_input)
		out = super().call(inputs)
		assert list(out.shape[1:]) == list(self.shape_image), "%s, %s"%(list(out.shape[1:]), self.shape_image)
		return out

	def get_config(self):
		return base.convert_config(self._config_param) 

class Decoder64(Decoder):
	def __init__(self, num_latents=10, activation=None, **kwargs):
		"""This is a decoder that takes in 64x64x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_image = [64,64,3]
		layer_params = ap.decoder64_architecture
		if activation is None:
			activation = ap.decoder64_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation, 
			**kwargs)

class Decoder256(Decoder):
	def __init__(self, num_latents=30, activation=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		shape_image = [256,256,3]
		layer_params = ap.decoder256_architecture
		if activation is None:
			activation = ap.decoder256_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation,
			**kwargs)

class Decoder512(Decoder):
	def __init__(self, num_latents=1024, activation=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		shape_image = [512,512,3]
		layer_params = ap.decoder512_architecture
		if activation is None:
			activation = ap.decoder512_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation,
			**kwargs)