# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:07:01 2019

@author: Mandar
"""

import tensorflow as tf


def iou(boxA, boxB):
  
  #evaluate the intersection points 
  xA = tf.maximum(boxA[:, 0], boxB[:, 0])
  yA = tf.maximum(boxA[:, 1], boxB[:, 1])

  xB = tf.minimum(boxA[:, 2], boxB[:, 2])
  yB = tf.minimum(boxA[:, 3], boxB[:, 3])

  # compute the area of intersection rectangle
  interArea = tf.maximum(0., xB - xA + 1) * tf.maximum(0., yB - yA + 1)

  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
  boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

  unionArea = (boxAArea + boxBArea - interArea)

  # return the intersection over union value
  return tf.reduce_mean(tf.reduce_mean(interArea / unionArea))