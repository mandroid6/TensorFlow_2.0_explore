# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:55:24 2019

@author: Mandar
"""

import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

def get_bb(file):
  
  #parse the annotations
  path = f'annotations/xmls/{file}'
  tree = ET.parse(path)
  root = tree.getroot()
  
  ob = root.find('object')
  bndbox = ob.find('bndbox')
  xmin = bndbox.find('xmin').text
  xmax = bndbox.find('xmax').text

  ymin = bndbox.find('ymin').text
  ymax = bndbox.find('ymax').text

  return((int(xmin), int(ymin)), (int(xmax), int(ymax)))
  
  
def draw_bb(file):
  #draw the bounding box
  img_path = f'images/{file[:-4]}.jpg'
  img = cv2.imread(img_path)
  
  (xmin, ymin), (xmax, ymax) = getBB(file)

  
  annotated = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
  
  plt.imshow(annotated[:,:,::-1])
  plt.axis('off')
  plt.show()