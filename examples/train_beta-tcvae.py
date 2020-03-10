"""This is an example for loading the celeba dataset, and training a beta-TCVAE model.

Attributes:
    batch_size (int): Batch Size
    dataset (dataset object): loads dataset when called.
    loss_func (TYPE): this is the defined mean squared error loss function
    model (TYPE): this is the keras model
"""

from utils import general_tools as gt 
import utils as ut
import tensorflow as tf
import numpy as np
import time
import cv2
import os
import shutil
# initialize model and dataset objects
dataset_manager, dataset = ut.dataset.get_celeba_data(
	ut.general_constants.datapath, 
	is_HD=False,
	group_num=8)
model = ut.tf_custom.architectures.variational_autoencoder.BetaTCVAE(
	1)


def preprocessing(inputs):
	# crop to 128x128 (centered), this number was experimentally found
	image_crop_size = [128,128]
	inputs=tf.image.crop_to_bounding_box(inputs, 
		(inputs.shape[-3]-image_crop_size[0])//2,
		(inputs.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	inputs = tf.image.convert_image_dtype(inputs, tf.float32)
	inputs = tf.image.resize(inputs, [64,64])
	return inputs

# run model and dataset objects
inputs_test, _ = dataset(2, False, True)

batch_size = 32
dataset = ut.dataset.DatasetBatch(dataset, batch_size).get_next

def preprocessed_data():
	inputs, _ = dataset()
	return preprocessing(inputs)


# reconstruction loss
class ImageMSE(tf.keras.losses.Loss):
	def call(self, actu, pred):
		reduction_axis = range(1,len(actu.shape))
		# per sample, should reduce sum on the image.
		loss = tf.math.reduce_sum(tf.math.squared_difference(actu, pred), reduction_axis)
		
		# per batch
		loss = tf.math.reduce_mean(loss)
		return loss

# regularization loss
def kld_loss_reduction(kld_loss):
	# per batch
	kld_loss = tf.math.reduce_mean(kld_loss)
	return kld_loss

# training
loss_func = ImageMSE()
optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.5)
total_metric = tf.keras.metrics.Mean()


import matplotlib.pyplot as plt

# load large data: (below is modeled off tensorflow website)
image_dir = "images"
model_setup_dir = "model_setup"
model_save_file = os.path.join(model_setup_dir, "model_weights.h5")



for i in [image_dir, model_setup_dir]:
	if os.path.exists(i):
		shutil.rmtree(i)
	os.mkdir(i)



step = -1

def save_image_step(step):
	steps = [1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
	return step in steps or step%5000 == 0

while 1: # set this using validation
	step+=1
	inputs = preprocessed_data()
	with tf.GradientTape() as tape:
		reconstruct = model(inputs)
		reconstruction_loss = loss_func(inputs, reconstruct)
		regularization_loss = kld_loss_reduction(model.losses[0])
		print(reconstruction_loss.numpy(), regularization_loss.numpy())
		loss = reconstruction_loss+regularization_loss

	grads = tape.gradient(loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))

	if save_image_step(step):
		print('step %s: mean loss = %s' % (
			step, 
			total_metric.result(),
			))
		true_inputs = preprocessing(inputs_test[:2])
		reconstruct_test = model(true_inputs).numpy()
		# concatenate the two reconstructions.
		a = np.concatenate((reconstruct_test[0], reconstruct_test[1]), axis=1)
		
		#concatenate the two true images
		b = np.concatenate((true_inputs[0], true_inputs[1]), axis=1)

		t_im = np.concatenate((a,b), axis=0)
		plt.imshow(t_im)
		plt.savefig(os.path.join(image_dir, "%d.png"%step))
	
	if step%10000 == 0:
		model.save_weights(model_save_file)

	#TBD: this should be replaced with validation
	if step>=100000:
		break