"""
This contains building blocks for the models, which would otherwise be somewhat
insufficient to be called a full implementation of a network

See the prebuilt_models package for more full implementations of models.
"""
import tensorflow as tf
from functools import reduce
import tensorflow.python.training.tracking.data_structures as td
try:
	#tensorflow 2.0
	from tensorflow.python.training.tracking.data_structures import _DictWrapper as DictWrapper
	from tensorflow.python.training.tracking.data_structures import ListWrapper as ListWrapper 
except ImportError:
	#tensorflow 1.14
	from tensorflow.python.training.tracking.data_structures import _DictWrapper as DictWrapper
	from tensorflow.python.training.tracking.data_structures import _ListWrapper as ListWrapper 


def _is_feed_forward(layer_param):
	"""
	check if a given list is feed forward.
	"""
	if type(layer_param) == ListWrapper: layer_param = list(layer_param)
	if not type(layer_param) == list or type(layer_param) == tuple: return False
	if not len(layer_param) == 1: return False
	if not type(layer_param[0]) == int: return False
	return True

def _is_conv2d(layer_param):
	"""
	check if a given list is for con2d, should not include upscale number
	"""
	if type(layer_param) == ListWrapper: layer_param = list(layer_param)
	if not type(layer_param) == list or type(layer_param) == tuple: return False
	if not len(layer_param) == 3: return False
	for i in range(len(layer_param)):
		if not type(layer_param[0]) == int: return False
	return True

class _Network(tf.keras.layers.Layer):
	"""
	General class for defining multiple layered structured networks.

	These networks will have: 
		- activations specified as a tf function or a dict which 
		specify the component layers.
		- layer_parameters as arguments to the function, each new layer must be a list
		of the relevant parameters.



	"""
	def __init__(self, activation, *layer_params):
		super().__init__()
		if type(activation) == dict:
			activation = {str(k):v for k,v in activation.items()}
		self._total_activations = activation 
		self._layer_params = list(layer_params)
		self._get_properties()


	def _apply_activation(self, layer_num):
		"""
		gets the correct activation for a given layer number. Starts at layer 0 to n-1
		"""
		activations = self._total_activations
		if type(activations) == dict or type(activations) == DictWrapper:
			if str(layer_num) in activations:
				current_activation = activations[str(layer_num)]
			elif str(layer_num - self._num_layers) in activations:
				current_activation = activations[str(layer_num - self._num_layers)]
			else:
				current_activation = activations["default"]
		else:
			current_activation = activations
		return current_activation

	def _get_properties(self):
		"""
		sets the properties of the layers.
		"""
		# layer properties
		for layer in self._layer_params:
			if type(layer) == ListWrapper: layer = list(layer)
			#layer parameters must be defined in a list
			assert type(layer) == list or type(
				layer) == ListWrapper, "layer parameters must be defined in a list or tuple"
		self._num_layers = len(self._layer_params)

		#activation properties
		if type(self._total_activations) == dict:
			assert "default" in self._total_activations

class ResnetBlock(_Network):
	"""
	Creates a convolutional resnet block given the parameters. 
	Convolutional layer specifications must be [depth, kernel, stride] 

	Activations must be a tensorflow activation function object, which will be the default 
	activation for all of the layers, or a dict
	with layer_number:tensorflow activation function. The dict must constain a "default"
	key, which will be used as the default activation, if not otherwise specified.

	Examples:
		>>> import numpy as np
		>>> inputs = np.random.normal(size=[32,64,64,3])
		>>> activation = tf.nn.relu #{"default":tf.nn.relu, -1:tf.math.sigmoid}
		>>> a = ResnetBlock( 
		>>> 	[32,1,1], # layer 1, for matching dimensions 
		>>> 	[32,3,1], # layer 2
		>>> 	[32,3,1], # layer 3
		>>> 	activation = activation
		>>> 	)
		>>> a(inputs) # the model must be run for keras to collect the trainable variables/weights
		>>> print(len(a.weights))

	Args:
		activation: This is the activation functions that will be performed on the network
		*layer_params: convolutional layers specifications in order.
	"""
	def __init__(self, *layer_params, activation=None, conv2d_obj=tf.keras.layers.Conv2D):
		assert not activation is None, "activation must be defined" 
		layer_params = list(layer_params)
		assert self.is_layers_valid(layer_params), "parameters specified do not match requirements for object."
		super().__init__(activation, *layer_params)
		self._is_skip_from_first_layer = self._layer_params[0][1] == 1
		# create the layers
		self._conv2d_layers = []
		for i in range(len(self._layer_params)):
			params = self._layer_params[i]
			conv2d_layer = conv2d_obj(
				*params, 
				padding="same", 
				activation=self._apply_activation(i))
			self._conv2d_layers.append(conv2d_layer)

	def call(self, inputs):
		initial_pred = inputs
		pred = initial_pred # so we call pipeline this in the loop
		for conv2d in self._conv2d_layers:
			pred = conv2d(pred)
			if self._is_skip_from_first_layer: #layers of size 1 will be used for matching dimensions, and the shortcut will start here.
				initial_pred = pred
		pred = pred + initial_pred
		return pred

	@staticmethod
	def is_layers_valid(layer_param):
		"""
		check if a given list is for resnet_block, should not include upscale number
		"""
		if type(layer_param) == ListWrapper: layer_param = list(layer_param)
		if not type(layer_param) == list or type(layer_param) == tuple: return False
		for i in layer_param:
			if not _is_conv2d(i): return False
		return True

class _ConvNetBase(_Network):
	"""
	Base class for CNNs. Accepts 3 types of layers: ResNet, Conv2D/Conv2DTranspose,
	and Dense. These layers will be checked for in the is_which_layer below.
	"""
	def create_conv2d_layers(self, layer_p, layer_num, conv2d_obj=tf.keras.layers.Conv2D):
		conv2d_layer = [] # local layer items
		layer_type = self.is_which_layer(layer_p, is_separate=False)
		assert layer_type > 1, "%d, %s"%(layer_type, layer_p)
		if layer_type == 2:
			conv2d_layer.append(conv2d_obj(
								*layer_p, 
								padding="same", 
								activation=self._apply_activation(layer_num)))
		elif layer_type == 3:
			conv2d_layer.append(ResnetBlock(
								*layer_p,
								activation=self._apply_activation(layer_num),
								conv2d_obj=conv2d_obj
								))
		else:
			raise Exception("Unknown layer type, %d"%layer_type)

		return conv2d_layer

	def create_ff_layers(self, layer_p, layer_num):
		ff_layer = [] # local layer items
		layer_type = self.is_which_layer(layer_p)
		assert layer_type == 1
		ff_layer.append(tf.keras.layers.Dense(
			*layer_p,
			activation=self._apply_activation(layer_num)))
		return ff_layer

	@staticmethod
	def separate_ff_and_conv(layer_params):
		ff_layer_params = []
		conv_layer_params = []
		for layer_p in layer_params:
			layer_type = _ConvNetBase.is_which_layer(layer_p)
			assert layer_type
			if layer_type == 1:
				ff_layer_params.append(layer_p)
			elif layer_type == 2 or layer_type == 3:
				conv_layer_params.append(layer_p)
			else:
				raise Exception("Unknown seperation value")
		return conv_layer_params, ff_layer_params

	@staticmethod
	def separate_upscale_or_pooling_parameter(layer_params):
		"""
		Takes in parameters for a single layer or resnet block

		separates upscale or pooling parameter.
		"""
		return layer_params[:-1], layer_params[-1]

	@staticmethod
	def is_layers_valid(layer_params):
		if type(layer_params) == ListWrapper: layer_params = list(layer_params)
		if not type(layer_params) == list or type(layer_params) == tuple: return False
		for i in layer_params:
			if not _ConvNetBase.is_which_layer(i): return False
		return True

	@staticmethod
	def is_which_layer(layer_param, is_separate=True):
		if type(layer_param) == ListWrapper: layer_param = list(layer_param)
		if _is_feed_forward(layer_param):
			return 1
		else:
			if is_separate:
				layer_param, _ = _ConvNetBase.separate_upscale_or_pooling_parameter(layer_param)
			if  _is_conv2d(layer_param):
				return 2
			elif ResnetBlock.is_layers_valid(layer_param):
				return 3
		return 0

class ConvolutionalNeuralNetwork(_ConvNetBase):
	"""
	This is a convolutional neural network class, used to build 
	CNNs, can be used as an encoder in a VAE, or a discriminator in a GAN
	
	Will follow the structure of: resnets, cnns -> feed forward NNs
	resnets and cnns can be mixed inbetween, but the feed forwards must be at the end 
	and will apply average pooling where specified
	pooling will be done after the layer.

	Activations must be a tensorflow activation function object, which will be the default 
	activation for all of the layers, or a dict
	with layer_number:appropriate activation specification. The dict must constain a "default"
	key, which will be used as the default activation, if not otherwise specified.
	
	appropriate activation specification is the allowed activation for that type of layer.
	Resnet can be a dict, or a tensorflow operation function.
	conv2d and feed forwards requires a tensorflow operation function.

	Resnets should be defined by a list of lists
	CNNs should be defined by a list of three ints.
	Feed forward should be a list of one int.

	Resnets and CNNs can have an int at the end of the list to signify 
	pooling

	Examples:
		>>> import numpy as np
		>>> inputs = np.random.randint(0,255,size=[8,512,512,3], dtype=np.uint8).astype(np.float32)
		>>> activations = {"default":tf.nn.relu, -1:lambda x: x}
		>>> a = ConvolutionalNeuralNetwork(*[
		>>> 	[16,5,1,2],  # conv layer, last element is amount of pooling, for no pooling, put None.
		>>> 	[[32,1,1], [32,3,1], [32,3,1], 2], #resnet block, last element is amount of pooling, for no pooling, put None.
		>>> 	[[64,1,1], [64,3,1], [64,3,1], 2], 
		>>> 	[[128,1,1], [128,3,1], [128,3,1], 2], 
		>>> 	[[256,1,1], [256,3,1], [256,3,1], 2], 
		>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
		>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
		>>> 	[[256,3,1], [256,3,1], 2], 
		>>> 	[[256,3,1], [256,3,1], 1], 
		>>> 	[4096], # this is the number of latent elements
		>>> 	#[1024]
		>>> 	], activation=activations, input_shape=[512,512,3])
		>>> print(a.shape_before_flatten)
		>>> print(a(inputs).shape) # the model must be run for keras to collect the trainable variables/weights
		>>> print(len(a.weights))
	Args:
		activations: This is the activation functions that will be performed on the network
		input_shape: This is the shape of the input, not including the batch size.
		*layer_params: convolutional layers specifications in order.
	"""
	def __init__(self, *layer_params, activation=None, shape_input=None):
		assert not activation is None, "activation must be defined" 
		assert not shape_input is None, "input_shape must be defined" 
		self._shape_before_flatten = None
		layer_params = list(layer_params)
		assert self.is_layers_valid(layer_params), "parameters specified do not match requirements for object." 
		self.conv_layer_params, self.ff_layer_params = self.separate_ff_and_conv(layer_params)
		super().__init__(activation, *layer_params)
		assert list(self._layer_params) == self.conv_layer_params+self.ff_layer_params, "conv and resnet layers parameters must be first."
			#make sure that the layers have been defined in order.
			#if you want to automate the layers to rearrange themselves, make sure to 
			#take care of the activations as well.
		self.shape_input=shape_input
		self._create_model()

	def _create_model(self):
		self.layer_objects = []
		
		# add the Conv and ResNet layers
		conv_layer = [tf.keras.layers.InputLayer(self.shape_input)]
		for i in range(len(self.conv_layer_params)):
			#setup parameters
			layer_p, up_param = self.separate_upscale_or_pooling_parameter(self.conv_layer_params[i])

			# get the convolution portion
			conv_layer+=self.create_conv2d_layers(layer_p, i)

			# get the average pooling portion
			if not up_param is None:
				conv_layer.append(tf.keras.layers.AveragePooling2D(
				up_param, padding="valid"))

		conv_layer = tf.keras.Sequential(conv_layer, "convolutional_layers_block")
		self.layer_objects.append(conv_layer)
		self._shape_before_flatten = list(conv_layer.output_shape[1:])

		ff_layer = []
		for i in range(len(self.ff_layer_params)):
			# setup parameters
			layer_num = len(self.conv_layer_params)+i

			# flattening first
			if not i:
				ff_layer.append(tf.keras.layers.Flatten())
			
			# get the ff layers 
			ff_layer+=self.create_ff_layers(self.ff_layer_params[i], layer_num)
			
		ff_layer = tf.keras.Sequential(ff_layer, "feed_forward_layers_block")
		self.layer_objects.append(ff_layer)

	def call(self, inputs):
		pred = inputs
		pred = self.layer_objects[0](pred)
		pred = self.layer_objects[1](pred)

		return pred

	@property
	def shape_before_flatten(self):
		assert not self._shape_before_flatten is None
		return list(self._shape_before_flatten)

class DeconvolutionalNeuralNetwork(_ConvNetBase):
	"""
	This is a deconvolutional neural network class, used to build 
	CNNs, can be used as an decoder in a VAE, or a discriminator in a GAN
	
	Will follow the structure of: resnets, feed forward NNs -> cnns 
	resnets and cnns can be mixed inbetween, but the feed forwards must be at the end 
	and will apply average pooling where specified
	upscaling will be done before the layer.

	Activations must be a tensorflow activation function object, which will be the default 
	activation for all of the layers, or a dict
	with layer_number:appropriate activation specification. The dict must constain a "default"
	key, which will be used as the default activation, if not otherwise specified.
	
	appropriate activation specification is the allowed activation for that type of layer.
	Resnet can be a dict, or a tensorflow operation function.
	conv2d and feed forwards requires a tensorflow operation function.

	Resnets should be defined by a list of lists
	CNNs should be defined by a list of three ints.
	Feed forward should be a list of one int.

	Resnets and CNNs can have an int at the end of the list to signify 
	upscaling.

	Reshaping will include a dense layer and a reshape layer. The dense layer
	will convert the input vector to the shape suitable to be reshaped, This will
	be applied after the ff networks and before the conv nets section. 
	No activations will be applied.

	Examples:
		>>> import numpy as np
		>>> latent = np.random.normal(size=[8,4096]).astype(np.float32)
		>>> activations = {"default":tf.nn.relu, -1:lambda x: x}
		>>> layer_params = [
		>>> 	[16,5,1,2], # conv layer, last element is amount of upscaling, for no upscaling, put None.
		>>> 	[[32,1,1], [32,3,1], [32,3,1], 2], # resnet block, last element is amount of pooling, for no pooling, put None.
		>>> 	[[64,1,1], [64,3,1], [64,3,1], 2], 
		>>> 	[[128,1,1], [128,3,1], [128,3,1], 2], 
		>>> 	[[256,1,1], [256,3,1], [256,3,1], 2], 
		>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
		>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
		>>> 	[[256,3,1], [256,3,1], 2], 
		>>> 	[[256,3,1], [256,3,1], 1], 
		>>> 	[4096], # this is the number of latent elements
		>>> 	#[1024]
		>>> 	]
		>>> layer_params = layer_params[::-1]
		>>> layer_params[-1][0] = 3 #num channels
		>>> decoder = DeconvolutionalNeuralNetwork(*layer_params, activation=activations,shape_before_flatten=[8,8,256])
		>>> 
		>>> recon = decoder(latent)
		>>> print(recon)

	Args:
		activations: This is the activation functions that will be performed on the network
		*layer_params: convolutional layers specifications in order.
	"""
	def __init__(self, *layer_params, activation=None, shape_before_flatten=None):
		assert not activation is None, "activation must be defined" 
		assert not shape_before_flatten is None, "shape_before_flatten must be defined" 
		self.layer_objects_reshape = None
		layer_params = list(layer_params)
		assert self.is_layers_valid(layer_params), "parameters specified do not match requirements for object." 
		super().__init__(activation, *layer_params)
		self.conv_layer_params, self.ff_layer_params = self.separate_ff_and_conv(layer_params)
		assert list(self._layer_params) == self.ff_layer_params + self.conv_layer_params, "conv and resnet layers parameters must be first."
			#make sure that the layers have been defined in order.
			#if you want to automate the layers to rearrange themselves, make sure to 
			#take care of the activations as well.
		self.shape_before_flatten = shape_before_flatten
		self._create_model()

	def _create_model(self):
		self.layer_objects = []
		ff_layer = []
		# get the ff layers 
		for i in range(len(self.ff_layer_params)):
			ff_layer+=self.create_ff_layers(self.ff_layer_params[i], i)
		ff_layer = tf.keras.Sequential(ff_layer, "feed_forward_layers_block")			
		self.layer_objects.append(ff_layer)

		# add the Conv and ResNet layers
		conv_layer = []
		for i in range(len(self.conv_layer_params)):

			#setup parameters
			layer_p, up_param = self.separate_upscale_or_pooling_parameter(self.conv_layer_params[i])
			layer_num = len(self.ff_layer_params)+i
			
			if not i:
				prod_reshape = reduce(lambda x,y: x*y, self.shape_before_flatten)
				conv_layer += [tf.keras.layers.Dense(prod_reshape),
						tf.keras.layers.Reshape(self.shape_before_flatten)]

			# get the average pooling portion
			if not up_param is None:
				conv_layer.append(tf.keras.layers.UpSampling2D(
				up_param, interpolation="nearest"))

			# get the convolution portion
			conv_layer+=self.create_conv2d_layers(layer_p, layer_num, 
				conv2d_obj=tf.keras.layers.Conv2DTranspose)

		conv_layer = tf.keras.Sequential(conv_layer, "convolutional_layers_block")
		self.layer_objects.append(conv_layer)


	def call(self, inputs):
		pred = inputs
		pred = self.layer_objects[0](pred)
		pred = self.layer_objects[1](pred)
		return pred

def main():
	# for testing.
		import numpy as np
		latent = np.random.normal(size=[8,1024]).astype(np.float32)
		activations = {"default":tf.nn.relu, -1:lambda x: x}
		layer_params = [
			[16,5,1,2], # conv layer, last element is amount of upscaling, for no upscaling, put None.
			[[32,1,1], [32,3,1], [32,3,1], 2], # resnet block, last element is amount of pooling, for no pooling, put None.
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
		layer_params = layer_params[::-1]
		layer_params[-1][0] = 3 #num channels
		decoder = DeconvolutionalNeuralNetwork(*layer_params, activation=activations,shape_before_flatten=[8,8,256])
		
		recon = decoder(latent)
		print(recon)


if __name__ == "__main__":
	main()