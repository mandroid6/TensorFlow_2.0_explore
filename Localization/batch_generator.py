# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 10:02:52 2019

@author: Mandar
"""
import numpy as np
from PIL import Image
import utils

def batch_generator(files, batch_size = 32, sz = (256, 256)):
  
  while True: 
    
    #extract a random batch 
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    
    
    for f in batch:
        img_path = f'images/{f[:-4]}.jpg'
        img = Image.open(img_path)
        w,h = img.size
        
        img = img.resize(sz)
        (xmin, ymin), (xmax, ymax) = utils.get_bb(f)
        
        img = np.array(img).astype('float32')
        if len(img.shape) == 2:
          img = np.stack((img,)*3, axis=-1)

        else:
          img = img[:,:,0:3]
        
        box = np.array([xmin/w, ymin/h, xmax/w, ymax/h])

        batch_x.append(img/255)
        batch_y.append(box)

    #preprocess a batch of images and masks 
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    yield (batch_x, batch_y)