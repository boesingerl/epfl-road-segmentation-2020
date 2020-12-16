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

TRAIN = False
if TRAIN:
    #Parameters to load the data
    input_dir = "training/images"
    target_dir = "training/groundtruth/"
    img_size = (400, 400) #Don't edit size
    batch_size = 100
    seed = 10

    #Don't apply any data augmentation here, we just load the pictures in RAM
    grayscale_x = False
    crop_length = None 
    aug_gen_args = {
        'rotation_range' : 90,
        'width_shift_range' : 0.2,
        'height_shift_range' : 0.2,
        'horizontal_flip' : False,
        'vertical_flip' : False,
    }
    #Load the data (80 training images, 20 validation images)
    imgen = ImageGenerator(input_dir, target_dir, aug_gen_args, seed, img_size, batch_size, grayscale_x, force_batch_size=False)
    train_gen, _ = imgen.get_crop_generator(256, seed)


## #Load test images
print('Loading test set images...')
test_dir = "./test_set_images/"
test_imgs = load_test_images(test_dir)
print('Loading complete!')

## Predict

#Load the model
model_filename = 'level7-postprocessing.h5'
if not os.path.isfile(model_filename):
    model_url = 'https://www.dropbox.com/s/rtqze3353rulzfk/level7-postprocessing.h5?dl=1'
    print('Downloading Model...')
    wget.download(model_url)
    print('Download complete!\nLoading model...')
model = keras.models.load_model(model_filename)
print('Loading model complete!\nPredicting Images')

#Predict (+ postprocess ?) test images
predictions = [predict_from_image(model, x, stride=32) for x in tqdm(test_imgs,leave=False)]
thresholded = [(x > 127).astype(int)*255 for x in tqdm(predictions,leave=False)]
print('Finished predicting !')

#Save images in folder
os.mkdir('./preds')
save_predictions_to_folder("./preds/", thresholded)

#Use images saved in folder to create submission
submission_filename = 'final_submission.csv'
create_submission("./preds/", submission_filename)

print(f'Created submission {submission_filename} !')
