import numpy as np
from sklearn.metrics import f1_score

def middle_threshold(pred):
  min,max = np.min(pred),np.max(pred)
  return pred > ((max - min)/2)

def create_vanilla_threshold(threshold):
    return lambda pred : pred > threshold

def create_percentile_threshold(percentile):
  return lambda pred : pred > np.percentile(pred, percentile)
  
def compute_score_thresholded(threshold_func, preds, target):
    thresholded = np.stack([threshold_func(pred) for pred in preds])
    return f1_score(np.ravel(target.astype(int)), np.ravel(thresholded))
    
def select_best_threshold(threshold_funcs, preds, target):
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