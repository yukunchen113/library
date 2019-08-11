import numpy as np

def create_image_grid(images, aspect_ratio=[1,1]):
	#will create a 2D array of images to be as close to the specified aspect ratio as possible.
	#assumes that images will be able to cover the specified aspect ration min num images = aspect_ratio[1]*aspect_ratio[2]
	#only plots grayscale images right now (can be scaled to multi channel)
	num_images = len(images)
	
	#find the bounding box:
	bounding_box = np.asarray(aspect_ratio)
	while(1):
		#print(bounding_box)
		#print(np.prod(bounding_box), num_images)
		if np.prod(bounding_box) >= num_images:
			break
		bounding_box+=1

	final_image = np.zeros((bounding_box[0]*images.shape[1], bounding_box[1]*images.shape[2], images.shape[3]))
	#fill the available bounding box
	for i in range(num_images):
		row_num = i%bounding_box[0]*images.shape[1]
		col_num = i//bounding_box[0]%bounding_box[1]*images.shape[2]
		final_image[row_num:row_num+images.shape[1], col_num:col_num+images.shape[2]] = images[i,:,:,:]
	return np.squeeze(final_image)