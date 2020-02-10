import tensorflow as tf
simple64_layer_parameters = [
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[32,4,2, None],
	[256],
	[256]]
simple64_shape_before_flatten = [-1,4,4,32]

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

hq512_shape_before_flatten = [8,8,256]


default_activations = {"default":tf.nn.relu, -1:lambda x: x}