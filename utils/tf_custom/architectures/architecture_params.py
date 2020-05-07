import tensorflow as tf
import copy

simple64_shape_before_flatten = [4,4,32]
simple64_layer_parameters = [
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[256],
	[256]]


hq256_shape_before_flatten = [8,8,128]
hq256_layer_parameters = [
		[16,5,1,2], 
		[[32,1,1], [32,3,1], 2],
		[[64,1,1], [64,3,1], 2], 
		[[128,1,1], [128,3,1], 2], 
		[[128,3,1], [128,3,1], 2], 
		[1024],
		[512], # this is the number of latent elements
		]


hq512_shape_before_flatten = [8,8,256]
hq512_layer_parameters = [
		[16,5,1,2], 
		[[32,1,1], [32,3,1], [32,3,1], 2],
		[[64,1,1], [64,3,1], [64,3,1], 2], 
		[[128,1,1], [128,3,1], [128,3,1], 2], 
		[[256,1,1], [256,3,1], [256,3,1], 2], 
		#[[512,1,1], [512,3,1], [512,3,1], 2], 
		#[[512,1,1], [512,3,1], [512,3,1], 2], 
		[[256,3,1], [256,3,1], 2], 
		[[256,3,1], [256,3,1], 1], 
		[4096], # this is the number of latent elements
		#[1024]
		]

default_decoder_activations = {"default":tf.nn.leaky_relu, -1:tf.math.sigmoid}
default_encoder_activations = {"default":tf.nn.leaky_relu, -1:tf.keras.activations.linear}



