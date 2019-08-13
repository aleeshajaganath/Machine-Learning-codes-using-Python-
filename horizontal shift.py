# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:05:47 2019

@author: user
"""

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('G:/dataaugmentation/bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
'Values less than 1.0 darken the image, e.g. [0.5, 1.0], whereas '
'values larger than 1.0 brighten the image, e.g. [1.0, 1.5], where 1.0 has no effect on brightness.'
#datagen = ImageDataGenerator(brightness_range=[0.2,1.0])

' zoom_range[1-value, 1+value]'
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# create image data augmentation generator
#datagen = ImageDataGenerator(width_shift_range=[0,200])
#datagen = ImageDataGenerator(height_shift_range=0.3)
#datagen = ImageDataGenerator(horizontal_flip=True)
#datagen = ImageDataGenerator(rotation_range=90)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()