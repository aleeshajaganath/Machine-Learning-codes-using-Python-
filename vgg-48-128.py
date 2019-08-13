#
# Step 1: download and format the data.
#
# I use the Quilt T4 Python API to do this. `t4` adds a lot of features on top
# of Amazon S3 buckets that make them more useful to data scientists. But you
# can also use either raw `boto3` or the `aws` CLI package, if so inclined.
#

import numpy as np
import pandas as pd
import t4


img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'

# `browse` downloads the package manifest without downloading the images.
open_fruits = t4.Package.browse('quilt/open_fruit', 's3://quilt-example')

# `fetch` lets us pull just the package assets we need to disk.
open_fruits['training_data/X_meta.csv'].fetch(metadata_filepath)
open_fruits['images_cropped'].fetch(img_dir)

# now we can get the X and y data we need for our generators (next step)
X_meta = pd.read_csv(metadata_filepath)
X = X_meta[['CroppedImageURL']].values
y = X_meta['LabelName'].values


#
# Step 2: define the data generators.
#
# Data generators are on-the-fly image transformers and are the recommended
# way of providing image data to models in Keras. They let you work with
# on-disk image data too large to fit all at once in-memory. And they allow
# you to preprocess the images your model sees with random image 
# transformations and standardizations, a key technique for improving model
# performance. To learn more, see https://keras.io/preprocessing/image/.
# 

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Our training data will use a wide assortment of transformations to try
# and squeeze as much variety as possible out of our image corpus.
# However, for the validation data, we'll apply just one transformation,
# rescaling, because we want our validation set to reflect "real world"
# performance.
#
# Also note that we are using an 80/20 train/validation split.
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)

# I found that a batch size of 128 offers the best trade-off between
# model training time and batch volatility.
batch_size = 128

# Notice the tiny target size, just 48x48!
train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#
# Step 3: define the model.
# 
# For the purposes of this article I based the core of my model on VGG16,
# a pretrained CNN architecture somewhat on the simpler side. This version
# of VGG16 is one trained on the famed ImageNet (http://www.image-net.org/)
# which includes some fruits in its list of classes, so performance should
# be decent. I add a new top layer consisting of a large-ish fully 
# connected layer with moderate regularization in the form of dropout.
# There are 12 output classes, so the output layer has 12 nodes.
#

prior = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=(48, 48, 3)
)
model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.1, name='Dropout_Regularization'))
model.add(Dense(12, activation='sigmoid', name='Output'))


# Freeze the VGG16 model, e.g. do not train any of its weights.
# We will just use it as-is.
for cnn_block_layer in model.layers[0].layers:
    cnn_block_layer.trainable = False
model.layers[0].trainable = False


# Compile the model. I found that RMSprop with the default learning
# weight worked fine.
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#
# Step 4: fit the model.
#
# Finally we fit the model. I use two callbacks here: EarlyStopping,
# which stops the model short of its full 20 epochs if validation 
# performance consistently gets worse; and ReduceLROnPlateau, which 
# reduces the learning rate 10x at a time when it detects model 
# performance is no longer improving between epochs.
#

# Recall that our dataset is highly imbalanced. We deal with this
# problem by generating class weights and passing them to the model
# at training time. The model will use the class weights to adjust
# how it trains so that each class is considered equally important to
# get right, even if the actual distribution of images is highly 
# variable.
import os
labels_count = dict()
for img_class in [ic for ic in os.listdir('images_cropped/') if ic[0] != '.']:
    labels_count[img_class] = len(os.listdir('images_cropped/' + img_class))
total_count = sum(labels_count.values())
class_weights = {cls: total_count / count for cls, count in 
                 enumerate(labels_count.values())}


model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2)
    ]
)