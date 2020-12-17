# -*- coding: utf-8 -*-
import seaborn as sns
import os
import numpy as np
import tensorflow as tf
import os.path
import wget
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tqdm import tqdm
from sklearn.metrics import f1_score
from libs.models import *
from libs.ImageGen import *
from libs.sliding_window import *
from libs.post_process import *
from libs.threshold import *
from libs.submission import *
from imgaug import augmenters as iaa
from tensorflow.keras.metrics import Recall, Precision

#Create the folder in which we save submission images
tmp_folder = './runpy_submissions/'
submission_filename = 'final_submission.csv'

if(os.path.isdir(tmp_folder)):
    print(f"Folder {tmp_folder} already exists, please delete it to use run.py")
    exit(1)

if(os.path.isfile(submission_filename)):
    printf(f"File {submission_filename} already exists, please delete it to use run.py")
    exit(1)
    
    
## #Load test images
print('Loading test set images...')
test_dir = "./test_set_images/"
test_imgs = load_test_images(test_dir)
print('Loading complete!')

TRAIN = False #Set to True if you wanna train, may need to restart to get better gpus on colab, and wait for a few hours
BEST_RESULT = False #Our justified results on postprocessing didnt lead to our best submission, toggle to True for our random-search submission
STRIDE = 32 #Our submissions were done using 32, set to higher values (that are divisors of 352, like 88,176 etc) for less prediction time, but possibly different results

if TRAIN:
    #Parameters to load the data
    input_dir = "training/images"
    target_dir = "training/groundtruth/"
    img_size = (400, 400) #Don't edit size
    batch_size = 100
    seed = 10
    
    #Augmentations for our ImageDataGenerator
    grayscale_x = False
    crop_length = None 
    aug_gen_args = {
        'rotation_range' : 90,
        'width_shift_range' : 0.2,
        'height_shift_range' : 0.2,
        'horizontal_flip' : False,
        'vertical_flip' : False,
    }
    
    #Load the data (All training no valid)
    print('Loading Training data...')
    imgen = ImageGenerator(input_dir, target_dir, aug_gen_args, seed, img_size, batch_size, grayscale_x, force_batch_size=False)
    train_gen, _ = imgen.get_crop_generator(256, seed)
    print('Done!')
    
    # Download the pretrained weights (need a weight initialization for this model to work)
    # These weights are only a small initialization of level 7 built without post processing for 3 epochs, 300 steps, just so it converges more quickly
    pretrained_w = 'lvl7_weights.h5'
    if not os.path.isfile(pretrained_w):
      print('Downloading init weights !')
      pretrained_url = 'https://www.dropbox.com/s/6pfh1dmyo4y7aqy/lvl7_weights.h5?dl=1'
      wget.download(pretrained_url)
      print('Done!')
    #Create our u-net using the pretrained weights
    model = unet(post_processing=True, levels=7,optimizer=Adam(lr=5e-4) ,metrics=[Recall(), Precision()], pretrained_weights=pretrained_w)

    #Fit for 20 epochs, we actually had to do this in two times (2*10 epochs), because colab disconnected us when running for too long
    model.fit(train_gen, epochs=20, steps_per_epoch=250)
else:
    #Load the model
    model_filename = 'level7-postprocessing.h5'
    if not os.path.isfile(model_filename):
        model_url = 'https://www.dropbox.com/s/rtqze3353rulzfk/level7-postprocessing.h5?dl=1'
        print('Downloading Model...')
        wget.download(model_url)
        print('Download complete!\nLoading model...')
    
    print('Loading model...\n')
    model = keras.models.load_model(model_filename, compile=False)
    print('Loading model complete!\nPredicting Images')


if BEST_RESULT:
    #Predictions performed on our random search
    predictions = [predict_from_image(model, x, stride=STRIDE) for x in tqdm(test_imgs,leave=False)]
    thresholded = [keep_large_area(((x > 80).astype(int)*255).astype('uint8'),500) for x in tqdm(predictions,leave=False)]
else:
    #Create predictions using rotated
    predictions = [predict_from_image_rotated(model, x, stride=STRIDE, rotations=[0,90,180,270], pad=False) for x in tqdm(test_imgs,leave=False)]
    #This threshold was selected in experiments with this model on train set, not included in run.py
    thresholded = [(x > 81.6).astype(int)*255 for x in tqdm(predictions,leave=False)]
print('Finished predicting !')

#Save images in folder
os.mkdir(tmp_folder)
save_predictions_to_folder(tmp_folder, thresholded)

#Use images saved in folder to create submission
create_submission(tmp_folder, submission_filename)

print(f'Created submission {submission_filename} !')
