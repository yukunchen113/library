import tensorflow as tf
import numpy as np

@tf.function
def cross_entropy(inputs, pred, epsilon=1e-7):
	"""
	Applys cross entropy per pixel between the inputs and the predictions

	we need to flatten our data, so we can reduce it per batch.

	Example:
		>>> a = np.random.normal(3,4,5)
		>>> b = np.random.normal(3,4,5)
		>>> res1 = cross_entropy(a,b).numpy()
	"""
	inputs = tf.keras.layers.Flatten()(inputs)
	pred = tf.keras.layers.Flatten()(pred)

	pred = tf.clip_by_value(pred, epsilon, 1-epsilon)
	loss = -tf.reduce_sum(
			inputs * tf.math.log(pred) + 
			(1-inputs) * tf.math.log(1-pred), 
			axis=1)
	return loss

@tf.function
def kl_divergence_with_normal(mean, log_var):
	"""
	gets kl divergence for a given men and variance, with the normal distribution
	log_var is the vector representing the diagonal of the covariance
	matrix. The covariances are assumed to be zero.

	Example:
		>>> a = np.random.normal(3,4,5)
		>>> b = np.random.normal(3,4,5)
		>>> res1 = kl_divergence_with_normal(a,b).numpy()
	"""
	return 0.5*(tf.math.exp(log_var)+tf.math.square(mean)-1-log_var)



