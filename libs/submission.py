import cv2
import os
import sys
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def load_test_images(folder):
    if os.path.isdir(folder):
        test_datagen = ImageDataGenerator()
        test_image_generator = test_datagen.flow_from_directory(folder, target_size=(608, 608), class_mode=None, shuffle=False, batch_size = 50)
        test_img = next(test_image_generator)
        numbers = [int(re.search(r"\d+", x.split("/")[-1]).group(0)) for x in test_image_generator.filenames]
        zipped = zip(test_img, numbers)
        return [k for k,v in sorted(zipped, key=lambda x: x[1])]
    else:
        raise RuntimeError("Path is not a folder")

def save_predictions_to_folder(folder, predictions):
    if os.path.isdir(folder):
        for pred,name in zip(predictions, range(1,1+len(predictions))):
            name = "{:03d}.png".format(name)
            cv2.imwrite(folder + name, pred)
    else:
        raise RuntimeError("Path is not a folder")
        
# assign a label to a patch
def patch_to_label(patch, foreground_threshold):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
        
def mask_to_submission_strings(image_filename,foreground_threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(image_filename.split("/")[-1].split(".")[0])
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, foreground_threshold)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename,foreground_threshold, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn,foreground_threshold))
            
           
def create_submission(folder, submission_filename,foreground_threshold=0.25):
    if os.path.isdir(folder):
        image_filenames = []
        for i in range(1, 51):
            image_filename = folder + "{:03d}.png".format(i)
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename,foreground_threshold, *image_filenames)
    else:
        raise RuntimeError("Path is not a folder")
        

# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def reconstruct_from_labels(label_file, image_id, h=16, w=16, nc=3, img_size=600.0):
    imgwidth = int(math.ceil((img_size/w))*w)
    imgheight = int(math.ceil((img_size/h))*h)
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)


    return Image.fromarray(im)
    
def plot_submission(label_file):
    fig, axs = plt.subplots(nrows=5,ncols=10,figsize=(20,10))
    for i in range(0,10):
      for j in range(0,5):
        img = reconstruct_from_labels(label_file,1 + i*5+j)
        axs[j,i].imshow(img)

