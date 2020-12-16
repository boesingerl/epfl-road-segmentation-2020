import tensorflow
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.util import view_as_blocks

def random_crop(img_x, img_y, random_crop_size, seed):

  np.random.seed(seed)    

  assert img_x.shape[:2] == img_y.shape[:2]
  height, width = img_x.shape[0], img_x.shape[1]
  dy, dx = random_crop_size
  
  x = np.random.randint(0, width - dx + 1)
  y = np.random.randint(0, height - dy + 1)
  
  cropped_x = np.zeros((dx, dx, img_x.shape[2]))
  cropped_y = np.zeros((dy, dy, img_y.shape[2]))
  
  cropped_x = img_x[y:y+dy, x:x+dx, :]
  cropped_y = img_y[y:y+dy, x:x+dx, :]
  return cropped_x, cropped_y

def crop_generator(xy_gen, crop_length, seed):
  """Take as input two zipped Keras ImageGen (Iterator) and generate random
  crops from the image batches generated by the original iterator.
  """
  while True:
      batch_x, batch_y = next(xy_gen)
      batch_x_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, batch_x.shape[3]))
      batch_y_crops = np.zeros((batch_y.shape[0], crop_length, crop_length, batch_y.shape[3]))
      
      for i in range(batch_x.shape[0]):
          batch_x_crops[i], batch_y_crops[i] = random_crop(batch_x[i], batch_y[i], (crop_length, crop_length), seed)
      yield batch_x_crops, batch_y_crops

def force_batch_size_gen(xy_gen, batch_size):
  while True:
    x,y = next(xy_gen)
    if(x.shape[0] == batch_size):
      yield x,y

    
def mask_to_block(mask, mask_size = (256,256), block_size = (16,16)):
  return view_as_blocks(mask.reshape(mask_size), block_size).mean(axis=(2,3))

def block_generator(xy_gen, threshold=0.25):
  while True:
    batch_x, batch_y = next(xy_gen)
    new_length = batch_y.shape[1]//16
    batch_ys = np.zeros((batch_y.shape[0], new_length, new_length))
    for i in range(batch_y.shape[0]):
      batch_ys[i] = (mask_to_block(batch_y[i]) > threshold).astype(int)
    yield batch_x, batch_ys
    
class ImageGenerator:
  def __init__(self,input_dir, target_dir, aug_gen_args, seed, input_img_size, batch_size, grayscale_x=True, force_batch_size=True):
    image_datagen = ImageDataGenerator(**aug_gen_args)
    mask_datagen = ImageDataGenerator(preprocessing_function=lambda x: (x>127).astype(int), **aug_gen_args)

    train_image_generator = image_datagen.flow_from_directory(input_dir,
                                                              class_mode=None,
                                                              seed=seed,
                                                              subset='training',
                                                              color_mode='grayscale' if grayscale_x==True else 'rgb',
                                                              target_size = input_img_size,
                                                              batch_size = batch_size)
    
    train_mask_generator = mask_datagen.flow_from_directory(target_dir,
                                                            class_mode=None,
                                                            seed=seed,
                                                            subset='training',
                                                            color_mode='grayscale',
                                                            target_size = input_img_size,
                                                            batch_size = batch_size)

    valid_image_generator = image_datagen.flow_from_directory(input_dir, 
                                                              class_mode=None,
                                                              seed=seed,
                                                              subset='validation',
                                                              color_mode='grayscale' if grayscale_x==True else 'rgb',
                                                              target_size = input_img_size,
                                                              batch_size = batch_size)
    
    valid_mask_generator = mask_datagen.flow_from_directory(target_dir,
                                                            class_mode=None,
                                                            seed=seed,
                                                            subset='validation',
                                                            color_mode='grayscale',
                                                            target_size = input_img_size,
                                                            batch_size = batch_size)

    self.train_generator = zip(train_image_generator, train_mask_generator)
    self.valid_generator = zip(valid_image_generator, valid_mask_generator)
    if force_batch_size:
        self.valid_generator = force_batch_size_gen(self.valid_generator,batch_size)
    

  def get_normal_generator(self):
    return self.train_generator, self.valid_generator

  def get_crop_generator(self, crop_length, seed):
    return crop_generator(self.train_generator, crop_length, seed), crop_generator(self.valid_generator, crop_length, seed)

  def get_block_generator(self, crop_length=None, crop_seed=None):
    if crop_length is not None:
      train,valid = self.get_crop_generator(crop_length, crop_seed)
    else:
      train,valid = self.get_normal_generator()
    return block_generator(train), block_generator(valid)