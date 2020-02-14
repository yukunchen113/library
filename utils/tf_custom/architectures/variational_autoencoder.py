import tensorflow as tf 
from utils.tf_custom.loss import kl_divergence_with_normal
from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.architectures import encoders as enc 
from utils.tf_custom.architectures import decoders as dec

class VariationalAutoencoder(tf.keras.Model):
	def __init__(self, name="variational_autoencoder", **kwargs):
		# default model
		super().__init__(name=name, **kwargs)
		self.create_encoder_decoder_64(**kwargs)

	def call(self, inputs):
		sample, mean, logvar = self.encoder(inputs)
		reconstruction = self.decoder(sample)
		self.add_loss(self.regularizer(sample, mean, logvar))
		return reconstruction

	def regularizer(self, sample, mean, logvar):
		return kl_divergence_with_normal(mean, logvar)
	
	def create_encoder_decoder_64(self, **kwargs):
		# default encoder decoder pair:
		self.encoder = enc.GaussianEncoder64(**kwargs)
		self.decoder = dec.Decoder64(**kwargs)

	def create_encoder_decoder_512(self, **kwargs):
		# default encoder decoder pair:
		self.encoder = enc.GaussianEncoder512(**kwargs)
		self.decoder = dec.Decoder512(**kwargs)

class BetaTCVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaTCVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.beta = beta

	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		kl_loss = kl_divergence_with_normal(mean, logvar)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)
		return tc + kl_loss

def main():
	import numpy as np
	import os
	batch_size = 8

	size = [batch_size,64,64,3]
	inputs = np.random.randint(0,255,size=size, dtype=np.uint8).astype(np.float32)
	a = BetaTCVAE(2)

	# test get reconstruction, only asserts shape
	assert a(inputs).shape == tuple(size)
	
	# test get vae losses
	assert len(a.losses) == 1

	# test model saving
	testdir = "test"
	if not os.path.exists(testdir):
		os.mkdir(testdir)
	testfile = os.path.join(testdir,"test.h5")
	w1 = a.get_weights()
	a.save_weights(testfile)

	# test model training
	a.compile(optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.MSE)
	a.fit(inputs, inputs, batch_size=batch_size, epochs=3)
	w3 = a.get_weights()

	# test model loading
	a.load_weights(testfile)
	w2 = a.get_weights()

	assert (w2[0] == w1[0]).all(), "weightsloading issue"
	assert (w3[0] == w2[0]).all(), "training weights updating issue"

if __name__ == '__main__':
	main()