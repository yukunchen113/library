import numpy as np
import tensorflow as tf
from utils.general_tools import Timer
class Traversal:
	
	def __init__(self, model, inputs):
		"""Traversal of the latent space class give one model and one set of inputs
		
		Examples:
			>>> def image_traversal(model, inputs, latent_of_focus=3, min_value=0, max_value=3, num_steps=15, is_visualizable=True)
			>>> 	traverse = Traversal(model, inputs)
			>>> 	traverse.traverse_latent_space(latent_of_focus=3, min_value=0, max_value=3, num_steps=15)
			>>> 	traverse.create_samples()
			>>> 	if not is_visualizable:
			>>> 		return traverse.samples
			>>> 	return traverse.construct_single_image()


		Args:
		    model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
		    inputs (numpy arr): Input images in NHWC
		    latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
		    min_value (int): min value for traversal
		    max_value (int): max value for traversal
		    num_steps (int): The number of steps between min and max value
		
		Returns:
		    Numpy arr: image
		"""
		self.model = model
		self.orig_inputs = inputs
		self.inputs = inputs
		self.samples = None
		self.latent_rep_trav = None #latent traversal to become shape [num traversal, N, num latents]

	def traverse_complete_latent_space(self, min_value=-3, max_value=3, num_steps=15):

		_, latent_rep, _ = self.model.encoder(self.inputs)
		latent_reps = []
		inputs = None
		
		# accumulate images for all the different latent representations, for all images
		for i in range(latent_rep.shape[1]):
			self.traverse_latent_space(latent_of_focus=i, 
				min_value=min_value, max_value=max_value, num_steps=num_steps)
			latent_reps.append(self.latent_rep_trav.copy())
			
			if inputs is None:
				inputs = np.empty((latent_rep.shape[1], *self.inputs.shape))
			inputs[i] = self.inputs

		# latents
		latent_reps = np.asarray(latent_reps)
		latent_reps = np.transpose(latent_reps, (2,0,1,3))
		self.latent_rep_trav = latent_reps.reshape((-1, *latent_reps.shape[-2:])).transpose((1,0,2))

		# inputs duplication
		inputs = np.transpose(inputs, (1,0,2,3,4))
		inputs = np.reshape(inputs, (-1, *inputs.shape[-3:]))
		self.inputs = inputs


	def traverse_latent_space(self, latent_of_focus=3, min_value=0, max_value=3, num_steps=15):
		"""traverses the latent space, focuses on one latent for each given image.
		
		Args:
		    latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
		    min_value (int): min value for traversal
		    max_value (int): max value for traversal
		    num_steps (int): The number of steps between min and max value
		
		"""
		t = Timer()
		# initialize latent representation of images
		_, latent_rep, latent_logvar = self.model.encoder(self.inputs)
		latent_rep = latent_rep.numpy()
		stddev = np.sqrt(np.exp(latent_logvar.numpy()[:,latent_of_focus]))

		# create latent traversal
		latent_rep_trav = []
		for i in np.linspace(min_value, max_value, num_steps):
			mod_latent_rep = latent_rep.copy()
			addition = np.zeros(mod_latent_rep.shape)
			addition[:,latent_of_focus] = i
			mod_latent_rep=latent_rep+addition
			latent_rep_trav.append(mod_latent_rep.copy())


		self.latent_rep_trav = np.asarray(latent_rep_trav)
		self.inputs = self.orig_inputs


	def create_samples(self):
		"""Creates the sample from the latent representation traversal
		"""
		assert not self.latent_rep_trav is None, "Please call traverse_latent_space first to get latent elements for self.latent_rep_trav"

		# flattened latent traversal for one batch dimension (assuming that the latent traversal is of the size, [num traversal, N, num latents])
		latent_rep = np.vstack(self.latent_rep_trav)

		# get the samples
		generated = self.model.decoder(latent_rep)

		# reshape back to [num traversal, N, W, H, C], as per self.latent_rep_trav
		self.samples = tf.reshape(generated, (*self.latent_rep_trav.shape[:2],*generated.shape[1:])).numpy()
		
	
	def construct_single_image(self):
		"""Contruct a single image to be displayed from samples. samples should be of shape [num traversal, N, W, H, C]
		
		Returns:
		    numpy array: array of images
		"""
		assert not self.latent_rep_trav is None, "Please call create_samples first to get sample to reconstruct"

		samples = np.concatenate(self.samples,-2) # concatenate horizontally
		samples = np.concatenate(samples,-3) # concatenate vertically
		real = np.concatenate(self.inputs,-3)

		image = np.concatenate((real, samples),-2) #concatenate the real and reconstructed images
		return image