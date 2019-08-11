import tensorflow as tf
import utils.tf_custom as tfc
from functools import reduce

class VariationalAutoEncoder(tf.keras.Model):
	"""
	This creates a variational auto encoder using tensorflow.
	Create the model and the respective
	"""
	# initialize the parameters
	def __init__(self, input_shape, encoder_layers, latent_size, loss_type, 
		decoder_layers = None,
		default_activation=tf.nn.relu, 
		latent_activation=lambda x: x, 
		output_activation=lambda x: x, **kwargs):
		"""
		Initializes the parameters, and constructs the vae.
		
		Regarding the specification method for the layers. Specification is in the form of a list of units.
		A unit can be a ResNet block, convolutional layer 

		Examples:
			>>> import numpy as np
			>>> 
			>>> #define the inputs
			>>> input_shape = [512,512,3]
			>>> inputs = np.random.randint(0,255,
			>>> 	size=[32]+input_shape).astype(np.float32)
			>>> 
			>>> #define the parameters
			>>> input_shape = input_shape
			>>> encoder_layers = [ # comment out more stuff if your computer can't handle this.
			>>> 	[16,5,1,2], 
			>>> 	[[32,1,1], [32,3,1], [32,3,1], 2],
			>>> 	[[64,1,1], [64,3,1], [64,3,1], 2], 
			>>> 	[[128,1,1], [128,3,1], [128,3,1], 2], 
			>>> 	[[256,1,1], [256,3,1], [256,3,1], 2], 
			>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
			>>> 	#[[512,1,1], [512,3,1], [512,3,1], 2], 
			>>> 	[[256,3,1], [256,3,1], 2], 
			>>> 	[[256,3,1], [256,3,1], None], 
			>>> 	[4096], 
			>>> 	#[1024]
			>>> 	]
			>>> latent_size = 512
			>>> loss_type = lambda inputs, prediction: tf.reduce_sum(
			>>> 	tf.losses.mean_squared_error(inputs, prediction,
			>>> 		reduction=tf.losses.Reduction.NONE), 
			>>> 	axis=list(range(1,len(inputs.shape))))
			>>> 
			>>> #make the model
			>>> a = VariationalAutoEncoder(
			>>> 	input_shape=input_shape,
			>>> 	encoder_layers=encoder_layers,
			>>> 	latent_size=latent_size,
			>>> 	loss_type=loss_type,
			>>> 		)
			>>> 
			>>> print(a(inputs).shape)
			>>> print(a.summary())
			>>> b,mu,logvar = a.encode(inputs, True)
			>>> print(b.shape)
			>>> print(a.decode(b).shape)


		Args:
			input_shape: input shape into the vae encoder.
			encoder_layers: layer specifications for the encoder, type: list. See specification method above
			latent_size: This is the size of the latent vector
			loss_type: function object (takes in tensorflow arrays) to apply to reconstruction loss (Eg. Mean squared error)
			decoder_layers: default is None. layer specifications for the decoder, if None, will reverse encoder. type: list, None. See specification method above
			default_activation: default is tf.nn.relu, The activation in the intermediate layers. Activaitons apply on the whole block, use dictionary for deeper specifications.
			latent_activation: default is no op, lambda x: x, Final activation to apply onto the latent output parameters. Eg. sigmoid for values constrained 0 to 1. Activaitons apply on the whole block, use dictionary for deeper specifications.
			output_activation: default is no op, lambda x: x, Final activation to apply onto the decoder output. Eg. sigmoid for values constrained 0 to 1. Activaitons apply on the whole block, use dictionary for deeper specifications.
		"""
		super().__init__()
		### Parameter Setup ###
		# encoder parameters
		self._encoder_layers = encoder_layers
		
		# latents parameters.
		self.latent_size = latent_size # latent layer size
		
		# decoder parameters
		if decoder_layers is None:
			# if decoder_layers is None, use encoder layers instead.
			self._decoder_layers = self._encoder_layers[::-1]
			self._decoder_layers[-1][0] = input_shape[-1]
		else:
			self._decoder_layers = decoder_layers
		
		# reconstruction loss parameters
		self._loss_type = loss_type
		
		# activation setup
		encoder_activation = {
			"default": default_activation,
			-1:latent_activation
			}

		decoder_activation = {
			"default": default_activation,
			-1:output_activation
			}

		### Model Definition ###
		# encodes up to the latent parameters
		encoder = tfc.model.ConvolutionalNeuralNetwork(
					*self._encoder_layers,
					activation=encoder_activation,
					input_shape=input_shape)
		self._encoder = tf.keras.Sequential([encoder,
					tf.keras.layers.Dense(self.latent_size*2)], "encoder")

		self._decoder = tfc.model.DeconvolutionalNeuralNetwork(
			*self._decoder_layers,
			activation=decoder_activation,
			shape_before_flatten=encoder.shape_before_flatten
			)

	def encode(self, inputs, return_dist=False):
		"""
		creates latents, if return_dist is True, return the distribution parameters 
		for the latent. inputs are a batch of images
		"""
		latent_mean, latent_log_var = tf.split(
			self._encoder(inputs), [self.latent_size, self.latent_size], -1)
		noise = tf.random.normal(tf.shape(latent_log_var))
		latent_output = tf.exp(0.5 * latent_log_var) * noise + latent_mean
		
		if return_dist:
			return latent_output, latent_mean, latent_log_var
		return latent_output

	def decode(self, inputs):
		"""
		creates images, inputs are a batch of latent vectors.

		returns image predictions
		"""
		pred = self._decoder(inputs)
		return pred

	def call(self, inputs):
		return self.decode(self.encode(inputs))

	def kl_isonormal_loss(self, latent_mean, latent_log_variance, reduce_sum = True):
		loss = tfc.loss.kl_divergence_with_normal(latent_mean, latent_log_variance)
		if reduce_sum:
			loss = tf.reduce_sum(loss, axis=1)
		return loss

	def reconstruction_loss(self, inputs, decoder_output):
		loss_type = self._loss_type
		return loss_type(inputs, decoder_output)

if __name__ == "__main__":
	import numpy as np

	#define the inputs
	input_shape = [512,512,3]
	inputs = np.random.randint(0,255,
		size=[32]+input_shape).astype(np.float32)
	#inputs = np.arange(32*512*512*3).reshape([32]+input_shape).astype(np.float32)

	#define the parameters
	input_shape = input_shape
	encoder_layers = [
		[16,5,1,2], 
		[[32,1,1], [32,3,1], [32,3,1], 2],
		[[64,1,1], [64,3,1], [64,3,1], 2], 
		[[128,1,1], [128,3,1], [128,3,1], 2], 
		[[256,1,1], [256,3,1], [256,3,1], 2], 
		#[[512,1,1], [512,3,1], [512,3,1], 2], 
		#[[512,1,1], [512,3,1], [512,3,1], 2], 
		[[256,3,1], [256,3,1], 2], 
		[[256,3,1], [256,3,1], None], 
		[4096], 
		#[1024]
		]
	latent_size = 512
	loss_type = lambda inputs, prediction: tf.reduce_sum(
		tf.losses.mean_squared_error(inputs, prediction,
			reduction=tf.losses.Reduction.NONE), 
		axis=list(range(1,len(inputs.shape))))
	
	#make the model
	a = VariationalAutoEncoder(
		input_shape=input_shape,
		encoder_layers=encoder_layers,
		latent_size=latent_size,
		loss_type=loss_type,
			)

	print(a(inputs).shape)
	print(a.summary())
	b,mu,logvar = a.encode(inputs, True)
	print(b.shape)
	print(a.decode(b).shape)
	#print(a.trainable_variables)
	a.save_weights("test/test", save_format="tf")

	b = VariationalAutoEncoder(
		input_shape=input_shape,
		encoder_layers=encoder_layers,
		latent_size=latent_size,
		loss_type=loss_type,
			)


	def test(vae, inputs):
		_, a, _ = vae.encode(inputs, True)
		return a

	print(np.all(test(a, inputs).numpy() == test(a, inputs).numpy()))
	print(np.all(test(a,inputs).numpy() == test(b,inputs).numpy()))
	b.load_weights("test/test")
	print(np.all(test(a,inputs).numpy() == test(b,inputs).numpy()))