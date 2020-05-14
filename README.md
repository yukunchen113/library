# library

This is my Machine Learning Library! This library includes custom functionalities for:
- machine learing models, 
- dataset loading
- visualizations.

Check out examples/ for some example code on how to use this library.

To contribute, make sure that examples/ contains some example code on how to run your code.

utils/ contains all the functions and classes that can be used

## Environment used and tested on:
- This code is mainly made for Tensorflow 2
- python 3.7

## To use this library:
- Add this library to your PYTHONPATH

## utils/:
- check out tf_custom for custom machine learning models and losses

### dataset.py
To add a dataset with pre-existing loading functionalities:
1. define your datapath. See utils/general_constants.py, overwrite datapath.
2. Format the datasets 
	- celeba:
		- make sure it is named _celeba_ in the your datapath and contains:
			- _images_
			- _list_bbox_celeba.txt_
			- _identity_CelebA.txt_
			- _list_landmarks_align_celeba.txt_
			- _list_attr_celeba.txt_
			- _list_landmarks_celeba.txt_

	- celeba-HQ:
		- make sure it is named celeba-hq in your datapath and contains:
			- _celeba-1024_ (contains only 1024x1024 images)
			- _celeba-512_ (contains only 512x512 images)
			- _celeba-256_ (contains only 256x256 images)
			- _celeba-128_ (contains only 128x128 images)
			- _celeba-64_ (contains only 64x64 images)
			- _list_bbox_celeba.txt_
			- _list_landmarks_align_celeba.txt_
			- _list_landmarks_celeba.txt_
			- _identity_CelebA.txt_
			- _list_attr_celeba.txt_
3. to make these all into valid hdf5 datasets, run examples/dataset_test.py

For an example on how to use the dataset see examples/dataset_test.py

To add in a custom dataset:
- put it into the datasets file. 
- For large datasets, use the GetData class which can retrieve images and labels and labels and save them into a .hdf5 file.

to use the built in celeba/celeba-hq datasets, see the comments in the get_celeba_dataset

If you want to load a dataset into a hdf5 file, you'll additionally need opencv. I only use openCV to quickly open an image to be loaded.


### utils/tf_custom/:
Check out the different custom tensorflow models and functions here!