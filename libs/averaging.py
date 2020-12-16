import numpy as np


def generate_F1_weights(f1_scores,cst_weight_prc=0.0) :
  weights = np.asarray([cst_weight_prc/len(f1_scores)]*len(f1_scores))
  f1_sum = np.sum(f1_scores)
  remaining_weight = 1.0 - cst_weight_prc
  f1_prc =  np.asarray([(f)/ f1_sum for f in f1_scores])
  w = weights + remaining_weight*f1_prc
  return w
  
def average_prediction(all_predictions,weights=None) :
  avg_preds = []
  for predictions in zip(*all_predictions) :
    predictions = np.asarray(predictions) 
    if weights is not None :
      avg_preds.append(np.sum(predictions*weights[:np.newaxis,np.newaxis,np.newaxis],axis=0))
    else :
      avg_preds.append(np.sum(predictions/len(predictions),axis=0))
  return np.asarray(avg_preds)

  