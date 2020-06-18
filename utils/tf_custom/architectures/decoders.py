"""This file contains the decoders that can be used in a VAE 
"""
from utils.tf_custom.architectures import base
import tensorflow as tf
from utils.tf_custom.architectures import architecture_params as ap
import copy
class Decoder(base.DeconvolutionalNeuralNetwork):
	def __init__(self, layer_params, num_latents, shape_image, activations=None, shape_before_flatten=None, **kwargs):
		"""Base class for decoders
		
		Args:
			layer_params (list): layer size specs
			shape_image (list): the shape of the input image (not including batch size)
			activations (dict): this is a dictionary of activations
		    shape_before_flatten (None, list): shape of activations before flattening.
		"""

		self.shape_image = shape_image
		self.shape_before_flatten = shape_before_flatten
		if activations is None:
			activations = ap.default_decoder_activations
	
		self._configuration_parameters = dict(
			layer_params=layer_params, 
			shape_before_flatten=shape_before_flatten, 
			shape_input=num_latents,
			#activation=activations, 
			)

		super().__init__(*layer_params, activation=activations, shape_before_flatten=shape_before_flatten, shape_input=num_latents)


	def call(self, latent_elements):
		out = super().call(latent_elements)
		assert list(out.shape[1:]) == list(self.shape_image), "%s, %s"%(list(out.shape[1:]), self.shape_image)
		return out

	def get_config(self):
		return {**base.convert_config(self._configuration_parameters), 
				"activations":base.convert_config(self._total_activations)} # activations are separately added 

class Decoder64(Decoder):
	def __init__(self, num_latents=10, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.shape_image = [64,64,3]
		self.layer_params = copy.deepcopy(ap.simple64_layer_parameters)[::-1]
		self.shape_before_flatten = ap.simple64_shape_before_flatten
		if "num_channels" in kwargs:
			self.shape_image[-1] = kwargs["num_channels"]
		out_layer = copy.deepcopy(self.layer_params[-1])
		out_layer[0] = self.shape_image[-1] # set num channels

		# add the image output layer
		self.layer_params.append(out_layer)

		# remove first conv spec
		del self.layer_params[len([i for i in self.layer_params if base.is_feed_forward(i)])]
		super().__init__(self.layer_params, 
			num_latents=num_latents, 
			shape_image=self.shape_image, 
			activations=activations, 
			shape_before_flatten=self.shape_before_flatten, **kwargs)

class Decoder256(Decoder):
	def __init__(self, num_latents=30, activations=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.shape_image = [256,256,3]
		self.layer_params = copy.deepcopy(ap.hq256_layer_parameters)[::-1]
		self.shape_before_flatten = ap.hq256_shape_before_flatten
		if "num_channels" in kwargs:
			self.shape_image[-1] = kwargs["num_channels"]
		self.layer_params[-1][0] = self.shape_image[-1] # set num channels
		super().__init__(self.layer_params, 
			num_latents=num_latents, 
			shape_image=self.shape_image, 
			activations=activations, 
			shape_before_flatten=self.shape_before_flatten, **kwargs)

class Decoder512(Decoder):
	def __init__(self, num_latents=1024, activations=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.shape_image = [512,512,3]
		self.layer_params = copy.deepcopy(ap.hq512_layer_parameters)[::-1]
		self.shape_before_flatten = ap.hq512_shape_before_flatten
		if "num_channels" in kwargs:
			self.shape_image[-1] = kwargs["num_channels"]
		self.layer_params[-1][0] = self.shape_image[-1] # set num channels
		super().__init__(self.layer_params, 
			num_latents=num_latents, 
			shape_image=self.shape_image, 
			activations=activations, 
			shape_before_flatten=self.shape_before_flatten, **kwargs)


def main():
	import numpy as np
	inputs = np.random.randint(0,255,size=[8,10], dtype=np.uint8).astype(np.float32)
	decoder64 = Decoder64()
	decoder64(inputs)

	inputs = np.random.randint(0,255,size=[8,1024], dtype=np.uint8).astype(np.float32)
	decoder512 = Decoder512()
	decoder512(inputs)



if __name__ == '__main__':
	save()