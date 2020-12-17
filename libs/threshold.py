import numpy as np
from sklearn.metrics import f1_score

def middle_threshold(pred):
    """Thresholds an image using the value between the min and max pixel values of the image"""
    min,max = np.min(pred),np.max(pred)
    return pred > ((max - min)/2)

def create_vanilla_threshold(threshold):
    """Creates a threshold function, which always thresholds an image with the given threshold value"""
    return lambda pred : pred > threshold

def create_percentile_threshold(percentile):
    """Creates a threshold function, which always thresholds an image with the percentile value (if a pixel is a higher value than the median for percentile = 0.5 for example)"""
    return lambda pred : pred > np.percentile(pred, percentile)
  
def compute_score_thresholded(threshold_func, preds, target):
    """Applies a threshold function to the predictions, and computes the f1 score with respect to the target images"""
    thresholded = np.stack([threshold_func(pred) for pred in preds])
    return f1_score(np.ravel(target.astype(int)), np.ravel(thresholded))
    
def select_best_threshold(threshold_funcs, preds, target):
    """Selects the best threshold with respect to f1-score from the given (threshold, threshold_func) pairs, predictions and target images"""
    assert len(preds) == len(target)
    best_func = None
    best_thresh = 0
    best_score = 0
    for threshold, threshold_func in threshold_funcs:
        score = compute_score_thresholded(threshold_func, preds, target)
        if score > best_score:
            best_thresh = threshold
            best_score = score
            best_func = threshold_func
    return best_func, best_thresh, best_score