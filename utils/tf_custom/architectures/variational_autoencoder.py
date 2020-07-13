import tensorflow as tf 
from utils.tf_custom.loss import kl_divergence_with_normal
from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.architectures import encoders as enc 
from utils.tf_custom.architectures import decoders as dec
class VariationalAutoencoder(tf.keras.Model):
	def __init__(self, name="variational_autoencoder", **kwargs):
		# default model
		super().__init__(name=name)
		self.create_default_vae(**kwargs)

	def create_default_vae(self, **kwargs):
		self.create_encoder_decoder_64(**kwargs)

	def _setup(self):
		# dynamically changing
		self.latest_sample = None
		self.latest_mean = None
		self.latest_logvar = None

	def call(self, inputs):
		self.latest_sample, self.latest_mean, self.latest_logvar = self.encoder(inputs)
		reconstruction = self.decoder(self.latest_sample)
		self.add_loss(self.regularizer(self.latest_sample, self.latest_mean, self.latest_logvar))
		return reconstruction

	def get_latent_space(self):
		return self.latest_sample, self.latest_mean, self.latest_logvar

	def regularizer(self, sample, mean, logvar):
		return kl_divergence_with_normal(mean, logvar)
	
	def create_encoder_decoder_64(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder64(**kwargs)
		self._decoder = dec.Decoder64(**kwargs)
		self._setup()

	def create_encoder_decoder_256(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder256(**kwargs)
		self._decoder = dec.Decoder256(**kwargs)
		self._setup()

	def create_encoder_decoder_512(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder512(**kwargs)
		self._decoder = dec.Decoder512(**kwargs)
		self._setup()
	
	def get_config(self):
		conf_params = {
			"encoder":self.encoder.get_config(),
			"decoder":self.decoder.get_config()}
		return conf_params

	@property
	def num_latents(self):
		return self.encoder.num_latents

	@property
	def shape_input(self):
		return self.encoder.shape_input

	@property
	def encoder(self):
		return self._encoder

	@property
	def decoder(self):
		return self._decoder


class BetaTCVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaTCVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.beta = beta

	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		kl_loss = kl_divergence_with_normal(mean, logvar)
		kl_loss = tf.reduce_sum(kl_loss,1)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)
		return tc + kl_loss
	
	def get_config(self):
		config_param = {
			**super().get_config(),
			"beta":self.beta}
		return config_param

class BetaVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.beta = beta

	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		kl_loss = self.beta*kl_divergence_with_normal(mean, logvar)
		kl_loss = tf.reduce_sum(kl_loss,1)
		return kl_loss
	
	def get_config(self):
		config_param = {
			**super().get_config(),
			"beta":self.beta}
		return config_param


def testall():
	import numpy as np
	import os
	import shutil
	batch_size = 8
	size = [batch_size,256,256,6]
	inputs = np.random.randint(0,255,size=size, dtype=np.uint8).astype(np.float32)
	a = VariationalAutoencoder(num_channels=6)
	#a = BetaTCVAE(2, num_channels=6)
	a.create_encoder_decoder_256(num_channels=6)

	# test get reconstruction, only asserts shape
	if not a(inputs).shape == tuple(size):
		print("input shape is different from size, change spec")
		custom_exit()
	
	# test get vae losses
	if not len(a.losses) == 1:
		print("regularizer loss not included")
		custom_exit()

	# test model saving
	testdir = "test"
	if not os.path.exists(testdir):
		os.mkdir(testdir)
	testfile1 = os.path.join(testdir, "test.h5")
	testfile2 = os.path.join(testdir, "test2.h5")
	w1 = a.get_weights()
	a.save_weights(testfile1)
	from inspect import signature
	print(a._updated_config())

	# test model training
	#model = 
	a.compile(optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.MSE)
	a.fit(inputs, np.random.randint(0,255,size=size, dtype=np.uint8).astype(np.float32), batch_size=batch_size, epochs=3)
	w3 = a.get_weights()

	# test model loading
	a.load_weights(testfile1)
	w2 = a.get_weights()

	a.summary()

	# test second model loading

	if not (w2[0] == w1[0]).all():
		print("weights loading issue")
		custom_exit(testdir)

	if (w3[0] == w1[0]).all():
		print("training weights updating issue")
		custom_exit(testdir)


	#tf.keras.backend.clear_session()
	c = BetaTCVAE(2, name="test")
	c.save_weights(testfile2)
	a.summary()


	print("Passed")
	#custom_exit(testdir)

def testload():
	import numpy as np
	import os
	import shutil
	testdir = "test"
	testfile = os.path.join(testdir, "test2.h5")
	a = BetaTCVAE(2, num_channels=6)
	a.load_weights(testfile)

	#"""
	import h5py
	f = h5py.File(testfile, "r")
	keys = list(f.keys())
	for i in keys:
		print(i)
		for k in f[i].keys():
			print(f[i][k]["kernel:0"])
		print()
	exit()
	print("Passed!")
	custom_exit(testdir)
	#"""

def custom_exit(files_to_remove=None):#roughly made exit to cleanup
	exit()



if __name__ == '__main__':
	testall()
	#testload()