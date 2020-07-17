import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import utils as ut
import utils.tf_custom.architecturesv2 as arc

def cprint(string):
	print('\033[94m'+string+'\033[0m')

########
# Base #
########
def ValidateParameters_test():
	LayerObj = arc.base.ValidateParameters
	def test_cases(LayerObj):
		check = LayerObj.check([])
		check = check and LayerObj.check([1,2,3,4])
		check = check and LayerObj.check(["test"])
		check = check and LayerObj.check([1.0])
		check = check and LayerObj.check([{"test":0}])
		return check
	assert test_cases(LayerObj), "basic checks failed"

	class NewLayerObj(LayerObj):
		@classmethod
		def additional_check(cls, layer_param, **kw):
			for i in layer_param:
				if not type(i) == int: return False 
			return layer_param
	assert NewLayerObj.check([1,2,3]) 
	assert NewLayerObj.check([])
	assert not test_cases(NewLayerObj), "additional check failed" 
	
	cprint("Passed ValidateParameters")

def Conv2D_test():
	layer_param = [3,1,2]
	layer = arc.base.Conv2D(*layer_param, padding="VALID", activation=tf.nn.leaky_relu, TEST="TEST")
	layer(np.ones((32,4,4,3)))
	layer = arc.base.Conv2DTranspose(*layer_param, padding="VALID", activation=tf.nn.leaky_relu)
	layer(np.ones((32,28,28,1)))
	cprint("Passed Conv2D")

def OptionWrapper_test():
	LayerObj = arc.base.BatchNormalization
	layer_param = ["bn"]
	if LayerObj.check(["ap", 2]):
		cprint("Failed OptionWrapper")
		return
	elif LayerObj.check([2]):
		cprint("Failed OptionWrapper")
		return
	layer = LayerObj(*layer_param)
	layer(np.ones((32,28,28,1)))

	LayerObj = arc.base.AveragePooling2D
	layer_param = ["ap", 2]
	layer = LayerObj(*layer_param)
	layer(np.ones((32,28,28,1)))

	LayerObj = arc.base.OptionWrapper(arc.base.Conv2D, identifier="test")
	layer_param = ["test", 3,1,2]
	layer = LayerObj(*layer_param, padding="VALID", activation=tf.nn.leaky_relu)
	layer(np.ones((32,4,4,3)))
	cprint("Passed OptionWrapper")


#########
# Block #
#########
def ConvBlock_test():
	inputs = np.random.normal(size=[32,64,64,3])
	activation = tf.nn.relu #{"default":tf.nn.relu, -1:tf.math.sigmoid}
	layer_param = [
				[32,1,1], # layer 1, for matching dimensions 
				[32,3,1], # layer 2
				[32,3,1] # layer 3
				]

	assert arc.block.ConvBlock.check(layer_param=layer_param, activation=activation)
	a = arc.block.ConvBlock( 
			*layer_param, 
			activation = activation
		)
	a(inputs) # the model must be run for keras to collect the trainable variables/weights
	#cprint([i.shape for i in a.get_weights()])
	cprint("Passed ConvBlock")

def ResnetBlock_test():
	inputs = np.random.normal(size=[32,64,64,3])
	activation = {"default":tf.nn.relu, 0:tf.keras.activations.linear, -1:tf.math.sigmoid}
	layer_param = [
				[32,1,1], # layer 1, for matching dimensions 
				[32,3,1], # layer 2
				[32,3,1] # layer 3
				]

	assert arc.block.ResnetBlock.check(layer_param=layer_param, activation=activation)
	a = arc.block.ResnetBlock( 
			*layer_param, 
			activation = activation
		)
	a(inputs) # the model must be run for keras to collect the trainable variables/weights
	#cprint([i.shape for i in a.get_weights()])
	cprint("Passed ResnetBlock")

def create_option_block_test():
	layer_param = [[3,1,2],["bn"]]
	conv2d_obj = arc.base.Conv2D
	conv2d_obj.default_kw = dict(padding="VALID")


	layer = arc.block.create_option_block(
			arc.base.Conv2D, 
			arc.base.BatchNormalization)(*layer_param, activation=tf.nn.leaky_relu)
	layer(np.ones((32,4,4,3)))
	cprint("Passed create_option_block")

###########
# Network #
###########

def NeuralNetwork_test():
	inputs = np.random.uniform(0,1, size=[32, 512])
	activation = {
		"default":tf.nn.relu, 
		-1:tf.keras.activations.linear}
	layer_param = [
		[[512],["bn"]],
		[[1024],["bn"]],
		[[512],["bn"]],
		[[256],["bn"]],
		[[128],["bn"]],
		[10]
		]
	a = arc.network.NeuralNetwork(*layer_param, activation=activation, shape_input=[512])
	assert a(inputs).shape == (32,10) # the model must be run for keras to collect the trainable variables/weights
	assert len(a.weights) == 32, str(len(a.weights))
	assert len(a.layers.layers) == len(layer_param)
	cprint("Passed NeuralNetwork")

def ConvolutionalNeuralNetwork_test():
	inputs = np.random.randint(0,255,size=[8,512,512,3], dtype=np.uint8).astype(np.float32)
	resnet_proj_activations = {"default":tf.nn.relu, 0:tf.keras.activations.linear}
	activation = {
		"default":tf.nn.relu, 
		1:resnet_proj_activations,
		2:resnet_proj_activations,
		3:resnet_proj_activations,
		4:resnet_proj_activations, 
		-1:tf.keras.activations.linear}
	layer_param = [
		[[16,5,1],  # conv layer
			["ap",2]], # pooling
		[[[[32,1,1],["bn"]], [32,3,1], ["bn"], [32,3,1]], #resnet block
			["ap",2]], # pooling
		[[[[64,1,1],["bn"]], [64,3,1], [["bn"], [64,3,1]]], ["ap",2]], 
		[[[128,1,1], [128,3,1], [128,3,1]], ["ap",2]], 
		[[[256,1,1], [256,3,1], [256,3,1]], ["ap",2]], 
		[[[256,3,1], [256,3,1]], ["ap",2]], 
		[[[256,3,1], [256,3,1]], ["ap",1]],
		[["flatten"],[4096]], # Dense
		]
	a = arc.network.ConvolutionalNeuralNetwork(*layer_param, activation=activation, shape_input=[512,512,3])
	assert a(inputs).shape == (8,4096) # the model must be run for keras to collect the trainable variables/weights
	assert len(a.weights) == 52, str(len(a.weights))
	assert len(a.layers.layers) == len(layer_param)
	cprint("Passed ConvolutionalNeuralNetwork")

def DeconvolutionalNeuralNetwork_test():
	inputs = np.random.randint(0,255,size=(8,4096), dtype=np.uint8).astype(np.float32)
	resnet_proj_activations = {"default":tf.nn.relu, 0:tf.keras.activations.linear}
	activation = {
		"default":tf.nn.relu, 
		4:resnet_proj_activations,
		5:resnet_proj_activations,
		6:resnet_proj_activations,
		7:resnet_proj_activations, 
		-1:tf.keras.activations.linear}
	layer_param = [
		[[4096]],
		[[8*8*256],["reshape", [8,8,256]]],
		[[[256,3,1], [256,3,1]], ["up",1]],
		[[[256,3,1], [256,3,1]], ["up",2]], 
		[[[256,1,1], [256,3,1], [256,3,1]], ["up",2]], 
		[[[128,1,1], [128,3,1], [128,3,1]], ["up",2]], 
		[[[[64,1,1],["bn"]], [64,3,1], [["bn"], [64,3,1]]], ["up",2]], 
		[[[[32,1,1],["bn"]], [32,3,1], ["bn"], [32,3,1]], ["up",2]], 
		[[3,5,1],["up",2]],
		]
	a = arc.network.DeconvolutionalNeuralNetwork(*layer_param, activation=activation, shape_input=[4096])
	#for i in a.layers.layers: print(i.output_shape)
	assert a(inputs).shape == (8,512,512,3) # the model must be run for keras to collect the trainable variables/weights
	assert len(a.weights) == 54, str(len(a.weights))
	assert len(a.layers.layers) == len(layer_param)
	cprint("Passed DeconvolutionalNeuralNetwork")

###########
# Encoder #
###########


###########
# Decoder #
###########


#######
# VAE #
#######



def test_all():
	ValidateParameters_test()
	OptionWrapper_test()
	Conv2D_test()

	ConvBlock_test()
	ResnetBlock_test()
	create_option_block_test()

	NeuralNetwork_test()
	ConvolutionalNeuralNetwork_test()
	DeconvolutionalNeuralNetwork_test()
