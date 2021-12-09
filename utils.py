import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=-1), tf.float32))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# def louge_l(real, pred):
#   preds = tf.cast(tf.argmax(pred, -1), tf.int64)
#   reals = tf.cast(real, tf.int64)
#   # preds = pred
#   mask_real = tf.math.logical_not(tf.math.equal(reals, 0))
#   mask_pred = tf.math.logical_not(tf.math.equal(preds, 0))
#   # print(preds, real)
#   real_m = tf.ragged.boolean_mask(reals, mask_real)
#   pred_m = tf.ragged.boolean_mask(preds, mask_pred)
#   loss_ = louge_object(real_m, pred_m)


#   return tf.math.reduce_mean(loss_[0])