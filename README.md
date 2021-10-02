# EPFL Road Segmentation 2020

<img src="https://user-images.githubusercontent.com/32189761/135729218-8829924c-c89b-490f-87e7-0befc8adcdf6.png" alt="Segmentation" height="200px"/>

This repository contains the code used to create our submission to the 2020 edition of the [EPFL Aicrowd Road Segmentation challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

In order to understand better what techniques we used, the best way is to read through the **Experiments.ipynb** notebook, which explains our usage of U-Nets, our selection of Data Augmentation, the Sliding Window technique we used to provide robustness, and different types of post processing methods attempted.

We placed 12th on the [Aicrowd leaderboard](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/leaderboards?challenge_round_id=695) ! This means that our solution definitely could use improvements, but we are satisfied of it.

## Libraries used and Execution Guide


<details>
  <summary>Click to show / hide</summary>
<br>
  
All the notebooks included were run using google colab, we thus recommend google colab for their execution. Should a local alternative be desirable, however the following versions of libraries, along with a version of python of 3.6.9 (the one present on google colab) are required:

```
imgaug==0.2.9
tensorflow==2.3.0
tensorflow-addons==0.8.3
tensorflow-datasets==4.0.1
tensorflow-estimator==2.3.0
tensorflow-gcs-config==2.3.0
tensorflow-hub==0.10.0
tensorflow-metadata==0.25.0
tensorflow-privacy==0.2.2
tensorflow-probability==0.11.0
Keras==2.4.3
matplotlib==3.2.2
seaborn==0.11.0
numpy==1.18.5
sklearn==0.22.2.post1
tqdm==4.41.1
```

Further, we cannot guarantee that any GPU can execute the same training as on colab, which contains a 16 GB GPU. Locally training a level 7 U-Net was an impossibility for us, as we ended up having OutOfMemory errors on local GPUs.

In order to run the run.py script, you will also need the wget library.

</details>

## Our usage of Google Colab

<details>
  <summary>Click to show / hide</summary>
<br>

In order to run all our experiments with good GPUs, we chose to use the Google Colab platform, thus, all our notebooks are hosted there. We also copied them to the github classroom for completeness (looking at code / outputs without running cells), but, since they all make use of google colab and google drive, to run them like we do, you need to follow these steps :

- Access this link that points to our Code Folder, named "Project-KLM" : https://drive.google.com/drive/folders/1FRLOodcV3puuPByXnAm0RzkGfGru2-EY?usp=sharing
- Add a shortcut to the Code Folder inside your root drive (Right-click on the folder, add a shortcut inside Drive), without changing the name
- When running a notebook, make sure that the Drive mount folder shows our code folder inside /content/gdrive/MyDrive, otherwise, the shortcut either has the wrong name, or is at the wrong location
- Sometimes colab allocates you worse GPUs than necessary, so you may need to reconnect to another machine if you try to train a model and get an OutOfMemory error when allocating Tensors.

Here is a description of everything in our Code Folder :
- archive : We kept most previous versions of our notebooks for completeness in this folder
- libs : All python libraries are kept under this folder (even those provided)
- models : All pretrained weights and models are in this folder
- submissions : We kept csv files for all important submissions in this folder.
- test_predictions : We kept image predictions for all important submissions in this folder
- test_set_images : The folder of test images
- training : The folder of training images
- validation_predictions : The image predictions that are done on our validation sets in our Experiments notebook are kept here.
- vis_postprocessing : Figures appearing in the report relating to postprocessing.
- ipynb files : All notebooks are described later
- run.py file: Same as the run.py in the github classroom folder, here for ease of use of the Running.ipynb notebook

</details>

## Documentation of our solution

<details>
  <summary>Click to show / hide</summary>
<br>
  
### The run.py file

The run.py performs the following steps :

- Downloading our best model
- Predicting the test images
- Creating a submission file

It uses tensorflow (and preferrably tensorflow-gpu, otherwise it is very slow) in order to run, and may not work with GPUs that have a lower amount of memory than the ones on colab.
It is also possible to train a model using the same parameters as our best model, instead of downloading the best model, but we again don't recommend it because it took us many hours to train it, on Colab GPUs. Keep in mind that even predicting may take some time due to our usage of sliding windows for prediction.

There are some parameters you can set in the run.py, they are :
- STRIDE : This parameter changes the stride parameter used in the sliding window, computation, it can be set to higher values (listed in the run.py file) for lesser computation time
- TRAIN : When set to true, trains a model from "scratch" (still need weight initalization), instead of downloading our best model
- BEST_RESULT : When set to true, doesn't use the best parameters we found based on our validation f1-score, but instead uses parameters we found when performing a random search on post-processing.

In order to run it using Colab, we provided you with a notebook called Running.ipynb in the aforementioned code folder. It simply installs the wget library and runs the run.py folder, to create the submission.

### The Experiments Notebook

The `Experiments.ipynb` notebook combines most of our experiments attempted on this project. 

Most cells have to be run on Google Colab or at least using similar/better GPUs (Nvidia K80 at least, but we can't guarantee that they didn't change since), although we don't even necessarily recommend running them, because training models can take multiple hours. Everything is already run, with shown output so that you can look at code and corresponding output.

It is divided into parts which are :

- Selecting Data Augmentation
- Selecting the level of our U-Net
- Best Input Size (and sliding window size)
- Best stride for our sliding window
- Trying out weighted loss
- Averaging models
- Post-Processing methods to use

### The "Model" Notebooks

These notebooks correspond to the models which we've considered to be noteworthy. They showcase how we train each model.

#### The Level 7 notebook

The level 7 notebook is most straightforward one, it simply showcases a normal level 7 model.

#### The AveragingModels notebook

This notebook showcases the training / loading of multiple level 5 models in order to average out the predictions, as an ensemble method.

#### The Weighted Level 7 notebook

This notebook showcases the training of a level 7 model which uses a weighted loss instead of binary cross entropy. That is, instead of giving equal weights to roads and background in the computation of the loss, we tried giving more or less weights, in order to tacke the class imbalance problem.

### The file libraries

In order to tidy up code inside the notebooks, we chose to move all shared / boilerplate code inside different python files which we use as libraries (listed under the libs folder).

#### image_gen.py

- random_crop
Given two images and a crop size, returns a tuple of cropped patches of given size, selected at the saame position in both images
- crop_generator
Given a generator of x,y samples, creates a generator that crops all images to the size given
- force_batch_size
Given a generator of x,y samples, creates a generator that returns only images that have the right batch size
- mask_to_block
Returns a view on the given mask as blocks, by averaging values in the mask
- block_generator
Given a generator of x,y samples, creates a generator that returns images as blocks, by using mask_to_block and a given threshold
- ImageGenerator
Given augmentation parameters, directories for input, and target images, and extra parameters, setups multiple generators, to be returned using the following methods :
  - get_normal_generator
    - Returns train and validation generators by appling augments
  - get_crop_generator
    - Returns a crop generator, applying crop_generator function on the get_normal_generator 
  - get_block_generator
    - Returns a crop generator, applying crop_generator function on the get_normal_generator or get_crop_generator (if given a crop_length)

#### models.py

- last_layers
Post processing CNN that fills segments
- unet
Returns a U-Net model from given parameters, with ability to specify input size, levels, pretrained weights, optimizer, callbacks, and using a post-processing CNN, using the last_layers function

#### sliding_windows.py

- windows_from_image
Returns window views on the given image, of size and stride given
- plot_windows
Used for debug, plots the provided windows
- image_from_windows
Recovers the image from the windows given (inverse of windows_from_image)
- pred_to_uint8
Converts float prediction to uint8
- predict_from_image
Gets window views on image, predicts roads pixel-wise all those windows, aggregates result of predictions back together to predict a complete image
- pad_border
Pads an image with zeros
- rotate_image
Applies affine rotation to images
- predict_from_image_rotated
Gets window views on image, predicts roads pixel-wise all those windows with applying given rotations, with or without padding depending on argument,and then aggregates result of predictions back together to predict a complete image

#### post_process.py

- morphological
Applies the given cv2 morphological operations to the image, using column, row, and square kernels sequentially
- open_pred
Applies the cv2.MORPH_OPEN operation to the image, using morphological
- close_pred
Applies the cv2.MORPH_CLOSE operation to the image, using morphological
- smooth_predictions
Applies Gaussian smoothing, for multiple steps in a row, with the given kernel size
- hough_find_lines
Returns all found lines using the Hough transform for lines and given parameters
- keep_large_area
Keeps only blobs of large enough area, with threshold as argument

#### submission.py

This file combines some of the provided functions, and adds other related to creating a submission.

- load_test_images
Loads all test images from the given directory
- save_predictions_to_folder
Saves all predictions (list of images) to given folder
- patch_to_label
Provided function : assign a label to a patch
- mask_to_submission_strings
Provided function : Reads a single image and outputs the strings that should go into the submission file
- masks_to_submission
Provided function : Converts images into a submission file
- create_submission
Creates a submission given ordered predictions
- binary_to_uint8
Converts binary labels to uint8
- reconstruct_from_labels
Provided function : Reconstruct images from label files
- plot_submission
Plots reconstructed submissions when provided a submission file, using reconstruct_from_labels

#### threshold.py

- middle_threshold
Thresholds an image using the value between the min and max pixel values of the image
- create_vanilla_threshold
Creates a threshold function, which always thresholds an image with the given threshold value
- create_percentile_threshold
Creates a threshold function, which always thresholds an image with the percentile value (if a pixel is a higher value than the median for percentile = 0.5 for example)
- compute_score_thresholded
Applies a threshold function to the predictions, and computes the f1 score with respect to the target images
- select_best_threshold
Selects the best threshold with respect to f1-score from the given (threshold, threshold_func) pairs, predictions and target images


#### averaging.py

- generate_F1_weights
Generates the custom weights for the average by giving more importance to the models with a higher F1 score
The importance of the F1_score can be customized by giving a minimum weight to all models with cst_weight_prc
- average_prediction
Takes a array of array of predictions and computes the mean item by item. The average can either be fair if no weights are set.
It can also be customized by passing it custom weights.


#### augmentation.py

- apply_augmentation
Augments the provided images by applying rotations of 0,90,180,270 degrees and flips

</details>

----

### Authors :

- Egli Marc
- Boesinger Leopaul
- Nejad Sattary Kamran
