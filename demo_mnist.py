#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network for mnist

demo for mnist

"""

import numpy as np
from skimage import io
import tensorflow as tf
import utils.mnist_net as mnet
import matplotlib.pyplot as plt
import sys
import configuration as conf

#-----------main

def readImage(filename):
    image =  io.imread(filename)
    image = image.astype(np.float32)
    image = 1.0 - image / 255.0    
    return image


if __name__ == '__main__':            
    if len(sys.argv) < 2:
        raise ValueError("device mode is required, use gpu or cpu")
    device_mode = sys.argv[1]    
    if  device_mode not in ["gpu", "cpu"] :
        raise ValueError("device not suppoerted, choose cpu or gpu")
        
    if device_mode == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
        
    label_files = ["./images/0.png", 
              "./images/1.png", 
              "./images/2.png", 
              "./images/3.png", 
              "./images/4.png", 
              "./images/5.png", 
              "./images/6.png", 
              "./images/7.png", 
              "./images/8.png", 
              "./images/9.png", 
              ]    
           
    labels = []
    for filename in label_files:
        labels.append(io.imread(filename))
    
    #filename  = "/media/hd_cvision/Datasets/Handwriting/MNIST/Test/digit_mnist_00033_3.png"
    

    with tf.device(device_name):
        net = mnet.net()    
    #to save snapshots    
    saver = tf.train.Saver()    
    #load mean
    #mean_img =np.fromfile(os.path.join(str_path, "mean.dat"), dtype=np.float32)
    #mean_img = np.reshape(mean_img, [28,28])    
    #mean_img = mean_img.astype(np.float32)    
    with tf.Session() as sess:
    #-------------------initialization of variable of graph
        saver.restore(sess, conf.SNAPSHOT_PREFIX + '-9999')    
        #the following is used only to make tensorboard available
        fig, xs = plt.subplots(1,2)
        while True:
            filename = input("image:")
            image = readImage(filename)
            y_pred = sess.run(net['y_pred'], feed_dict={net['x']: [image]})
            y_pred_cls = np.argmax(y_pred[0], 0)        
            prob = y_pred[0][int(y_pred_cls)]            
            for i in range(2):
                xs[i].axis("off")
            xs[0].imshow((1-image)*255, cmap='gray')
            xs[1].imshow(labels[int(y_pred_cls)], cmap='gray')
            xs[1].set_title(prob)
            plt.waitforbuttonpress()
    
    
