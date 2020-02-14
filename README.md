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
loads in the dataset, also provides functionality for saving the data. 

For an example on how to use the current dataset see examples/

To add in a custom dataset:
- put it into the datasets file. 
- For large datasets, use the GetData class which can retrieve images and labels and labels and save them into a .hdf5 file.

to use the built in celeba/celeba-hq datasets, see the comments in the get_celeba_dataset

If you want to load a dataset into a hdf5 file, you'll additionally need opencv. I only use openCV to quickly open an image to be loaded.


### utils/tf_custom/:
Check out the different custom tensorflow models and functions here!