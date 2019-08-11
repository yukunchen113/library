import tensorflow.compat.v1 as tf
def cross_entropy(inputs, pred, epsilon=1e-7):
	"""
	Applys cross entropy per pixel between the inputs and the predictions

	we need to flatten our data, so we can reduce it per batch.
	"""
	inputs = tf.layers.flatten(inputs)
	pred = tf.layers.flatten(pred)

	pred = tf.clip_by_value(pred, epsilon, 1-epsilon)
	return -tf.reduce_sum(
			inputs * tf.log(pred) + 
			(1-inputs) * tf.log(1-pred), 
			axis=1)

def kl_divergence(mean, log_var):
	return 0.5*(tf.exp(log_var)+tf.square(mean)-1-log_var)

def upscale2d(x, factor=2):
	##########
	#This function was taken from https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py and modified
	#This is used to reproduce the same upscaling as in the paper, modified to our purposes
	#please check out their work: PGGANs.
	##########
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	with tf.variable_scope('Upscale2D'):
		s = x.shape
		x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
		x = tf.tile(x, [1, 1, factor, 1, factor, 1])
		x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
	return x

def center_square_crop(images, dimensions):
	"""
	center crops the images to a square, with int dimensions.
	"""
	image_crop_size = [dimensions,dimensions]
	images=tf.image.crop_to_bounding_box(images, 
		(images.shape[-3]-image_crop_size[0])//2,
		(images.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	return images