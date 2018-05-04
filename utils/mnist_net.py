#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:29:43 2018

@author: jose.saavedra

Basic architecture for mnist
"""

import tensorflow as tf
import numpy as np
from . import layers

#mnist net
def net(input_shape = [None, 28, 28]):
    #placeholder for is_training = [False or True]. False is used for testing and True for training
    is_training = tf.placeholder(tf.bool,  shape=(), name = 'is_training')
    #placeholder for input data
    x = tf.placeholder(tf.float32, input_shape, name = 'x')
    #placeholder for labels    
    y_true = tf.placeholder(tf.float32, [None, 10], name = 'y_true')        
    #reshape input to fit a  4D tensor
    x_tensor = tf.reshape(x, [-1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], 1 ] )
    #conv_1    
    conv_1 = layers.conv_layer(x_tensor, shape = [3, 3, 1, 32], name='conv_1', is_training = is_training); 
    conv_1 = layers.max_pool_layer(conv_1, 3, 2) # 14 x 14
    print(" conv_1: {} ".format(conv_1.get_shape().as_list()))
    #conv_2
    conv_2 = layers.conv_layer(conv_1, shape = [3, 3, 32, 64], name = 'conv_2', is_training = is_training)
    conv_2 = layers.max_pool_layer(conv_2, 3, 2) # 7 x 7
    print(" conv_2: {} ".format(conv_2.get_shape().as_list()))
    #conv_3
    conv_3 = layers.conv_layer(conv_2, shape = [3, 3, 64, 64], name = 'conv_3', is_training = is_training)
    conv_3 = layers.max_pool_layer(conv_3, 3, 2) # 3 x 3
    print(" conv_3: {} ".format(conv_3.get_shape().as_list()))    
    #fully connected
    fc6 = layers.fc_layer(conv_3, 250, name = 'fc6')
    print(" fc6: {} ".format(fc6.get_shape().as_list()))
    #fully connected
    fc7 = layers.fc_layer(fc6, 10, name = 'fc7', use_relu = False)
    print(" fc7: {} ".format(fc7.get_shape().as_list()))    
    #gap = layers.gap_layer(conv_5) # 8x8
    #print(" gap: {} ".format(gap.get_shape().as_list()))    
    y_pred = tf.nn.softmax(fc7)
    #loss function-------------------------
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc7, labels = y_true)
    loss = tf.reduce_mean(cross_entropy)
    #---------------------------------------
    #accuracy
    y_pred_cls = tf.argmax(y_pred, 1)
    y_true_cls = tf.argmax(y_true, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls) , tf.float32))
    
    return {'is_training': is_training, 'x': x, 'y_true': y_true, 'y_pred': y_pred, 'loss': loss, 'acc': acc}
