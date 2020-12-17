import numpy as np


def generate_F1_weights(f1_scores,cst_weight_prc=0.0):
  """Generates the custom weights for the average by giving more importance to the models with a higher F1 score"""
  weights = np.asarray([cst_weight_prc/len(f1_scores)]*len(f1_scores))
  f1_sum = np.sum(f1_scores)
  remaining_weight = 1.0 - cst_weight_prc
  f1_prc =  np.asarray([(f)/ f1_sum for f in f1_scores])
  w = weights + remaining_weight*f1_prc
  return w
  
def average_prediction(all_predictions,weights=None):
  """Takes a array of array of predictions and computes the mean item by item. The average can either be fair if no weights are set. It can also be customized by passing it custom weights."""
  avg_preds = []
  for predictions in zip(*all_predictions) :
    predictions = np.asarray(predictions) 
    if weights is not None :
      avg_preds.append(np.sum(predictions*weights[:np.newaxis,np.newaxis,np.newaxis],axis=0))
    else :
      avg_preds.append(np.sum(predictions/len(predictions),axis=0))
  return np.asarray(avg_preds)

  