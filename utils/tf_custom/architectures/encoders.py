"""This file contains the encoders that can be used in a VAE or GAN. 
"""
import base
import tensorflow as tf
simple64_layer_parameters = [
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[256],
	[256]]

class GaussianEncoder64(base.ConvolutionalNeuralNetwork):
	shape_input = [64,64,3]
	def __init__(self, num_latents=10, activations=None):
		self.num_latents = num_latents
		if activations is None:
			activations = {"default":tf.nn.relu, -1:lambda x: x}
		layer_params = simple64_layer_parameters + [[self.num_latents*2]]
		super().__init__(*layer_params, activation=activations, shape_input=self.shape_input)
	
	def call(self,inputs):
		assert list(inputs.shape[1:]) == self.shape_input
		out = super().call(inputs)
		mean = inputs[:,:self.num_latents]
		logvar = inputs[:,self.num_latents:]
		return ##TBD

def main():
	import numpy as np
	inputs = np.random.randint(0,255,size=[8,64,64,3], dtype=np.uint8).astype(np.float32)
	encoder = GaussianEncoder64()
	print(encoder(inputs).shape)

if __name__ == '__main__':
	main()