import tensorflow as tf
from . import architecture_params as ap
from . import base
from . import network
class GaussianEncoder(network.ConvolutionalNeuralNetwork):
	def __init__(self, layer_params, num_latents, shape_input, activation, **kwargs):
		"""Base clase for a gaussian encoder
		
		Args:
		    layer_params (list): layer size specs
		    num_latents (int): the number of latent elements
		    shape_input (list): the shape of the input image (not including batch size)
		    activation (dict): this is a dictionary of activation
		    **kwargs: Description
		"""
		# add num latents
		layer_params = layer_params + [[num_latents*2]]

		# add additional attributes
		self.num_latents = num_latents

		super().__init__(*layer_params, activation=activation, shape_input=shape_input, **kwargs)
		
		# set the config
		self._config_param = dict(
					layer_params=layer_params, 
					shape_input=shape_input,
					num_latents=num_latents,
					activation=self.activation
					)
	
	def call(self,inputs):
		assert list(inputs.shape[1:]) == list(self.shape_input), "Invalid Input shape given: %s,  sepecified: %s"%(list(inputs.shape[1:]), self.shape_input)
		out = super().call(inputs)
		mean = out[:,:self.num_latents]
		logvar = out[:,self.num_latents:]
		sample = tf.exp(0.5*logvar)*tf.random.normal(
			tf.shape(logvar))+mean
		return sample, mean, logvar

	def get_config(self):
		return base.convert_config(self._config_param) # activation are separately added 

class GaussianEncoder64(GaussianEncoder):
	def __init__(self, num_latents=10, activation=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_input = [64,64,3]
		layer_params = ap.encoder64_architecture
		if activation is None:
			activation = ap.encoder64_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_input=shape_input, 
			activation=activation, 
			**kwargs)

class GaussianEncoder256(GaussianEncoder):
	def __init__(self, num_latents=30, activation=None, **kwargs):
		"""This is a gaussian encoder that takes in 512x512x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_input = [256,256,3]
		layer_params = ap.encoder256_architecture
		if activation is None:
			activation = ap.encoder256_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_input=shape_input, 
			activation=activation, 
			**kwargs)

class GaussianEncoder512(GaussianEncoder):
	def __init__(self, num_latents=1024, activation=None, **kwargs):
		"""This is a gaussian encoder that takes in 512x512x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_input = [512,512,3]
		layer_params = ap.encoder512_architecture
		if activation is None:
			activation = ap.encoder512_activations
		super().__init__(
			layer_params=layer_params, 
			num_latents=num_latents, 
			shape_input=shape_input, 
			activation=activation, 
			**kwargs)

def main():
	import numpy as np
	inputs = np.random.randint(0,255,size=[8,64,64,3], dtype=np.uint8).astype(np.float32)
	encoder = GaussianEncoder64()
	print(encoder(inputs)[0])
	print(encoder.shape_before_flatten)

if __name__ == '__main__':
	main()