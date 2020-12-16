# ML Project : Road Segmentation

## Libraries used and how to run

All the notebooks we created were run using google colab, we thus recommend to use it to interact with them. It should, however be equivalent to having the following versions of libraries :

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

We however cannot guarantee that any GPU can run the same training as on colab, for example, when we tried on our own computers to train a level 7 U-Net, we ended up having OutOfMemory errors on our GPUs.

In order to run the run.py script, you will also need the wget library.

## The Notebooks

The notebook combines most of the experiments we ever did on this project. 

Most cells have to be run on Google Colab or at least using the same/better GPUs (Nvidia K80 at least, but we can't guarantee that they didn't change since), but we don't even necessarily recommend running them, because training models can take multiple hours. 

It is divided into parts which are :

- Selecting Data Augmentation
- Selecting the level of our U-Net
- Different types of sliding windows
- Best stride for our sliding window
- Post-Processing methods to use
- Submitting

The end of the notebook provides the same functionality as the run.py file, i.e, downloading our best model, predicting, and creating a submission file. Keep in mind that even predicting may take some time due to our usage of sliding windows for prediction.

## The file libraries

In order to tidy up code inside the notebooks, we chose to move all shared / boilerplate code inside different python files which we use as libraries.

### image_gen.py

- random_crop
Given too images and a crop size, returns a tuple of cropped patches of given size, selected at the same position in both images
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

### models.py

- last_layers
Post processing CNN that fills segments
- unet
Returns a U-Net model from given parameters, with ability to specify input size, levels, pretrained weights, optimizer, callbacks, and using a post-processing CNN, using the last_layers function

### sliding_windows.py

- windows_from_image
Returns window views on the given image, of size and stride given
- plot_windows
Used for debug, plots the provided windows
- image_from_windows
Recovers the image from the windows given (inverse of windows_from_image)
- pred_to_uint8
Converts float prediction to uint8
- predict_from_image
Gets windows on image, predicts all those windows, aggregates result of preds back together to predict a complete image
- pad_border
Pads an image with zeros
- rotate_image
Applies affine rotation to images
- predict_from_image_rotated
Gets windows on image, predicts all those windows with applying given rotations, with or without padding depending on argument,and then aggregates result of preds back together to predict a complete image

### post_process.py

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

### submission.py

This file combines some of the provided functions, and adds other related to creating a submission.

- load_test_images
Loads all test images from the given directory
- save_predictions_to_folder
Provided function
- patch_to_label
Provided function
- mask_to_submission_strings
Provided function
- masks_to_submission
Provided function
- create_submission
Creates a submission given ordered predictions
- binary_to_uint8
Provided function
- reconstruct_from_labels
Provided function
- plot_submission
Plots reconstructed submissions when provided a submission file, using reconstruct_from_labels

### threshold.py

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

## The run.py file

The run.py performs the following steps :

- Downloading our best model
- Predicting the test images
- Creating a submission file

It is also possible to train a model using the same parameters as our best model, instead of downloading the best model, but we again don't recommend it because it took us many hours to train it, on Colab GPUs. Keep in mind that even predicting may take some time due to our usage of sliding windows for prediction.

----

### Authors :

- Nejad Sattary Kamran
- Egli Marc
- Boesinger LÃƒÆ’Ã‚Â©opaul
