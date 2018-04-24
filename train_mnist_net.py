#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network for mnist


"""

import numpy as np
import sys
import os
import utils.data as data
import utils.mnist_net as mnet
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import configuration as conf

if __name__ == '__main__':            
    if len(sys.argv) < 3:
        raise ValueError("input incorrect <mode> <device>")        
    run_mode = sys.argv[1]
    device_mode = sys.argv[2]        
    if  run_mode not in ["train", "test"] :
        raise ValueError("mode should be  train or test")       
    if  device_mode not in ["gpu", "cpu"] :
        raise ValueError("device not supported, choose cpu or gpu")
        
    if device_mode == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
           
    print ("loading data [train and test] \n")    
    filename_train = os.path.join(conf.DATA_DIR, "train.tfrecords")
    filename_test = os.path.join(conf.DATA_DIR, "test.tfrecords")
    #---------------read TFRecords data  for training
    data_train = tf.data.TFRecordDataset(filename_train)
    data_train = data_train.map(data.parser_tfrecord)
    data_train = data_train.batch(conf.BATCH_SIZE)    
    data_train = data_train.shuffle(conf.ESTIMATED_NUMBER_OF_BATCHES)        
    #---------------read TFRecords data  for validation
    data_test = tf.data.TFRecordDataset(filename_test)
    data_test = data_test.map(data.parser_tfrecord)
    data_test = data_test.batch(conf.BATCH_SIZE)    
    data_test = data_test.shuffle(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST)
    #defining saver to save snapshots
    #defining a reinitializable iterator                
    iterator = Iterator.from_structure(data_train.output_types, data_train.output_shapes)    
    iterator_test = Iterator.from_structure(data_test.output_types, data_test.output_shapes)    
    
    next_batch = iterator.get_next()
    next_batch_test = iterator_test.get_next()
    #tensor that initialize the iterator:
    training_init_op = iterator.make_initializer(data_train)    
    testing_init_op = iterator_test.make_initializer(data_test)    
    print ("OK")
    with tf.device(device_name):
        net = mnet.net()    
    print("train")
    #to save snapshots    
    saver = tf.train.Saver()    
    #load mean
    #mean_img =np.fromfile(os.path.join(DATA_DIR, "mean.dat"), dtype=np.float32)
    #mean_img = np.reshape(mean_img, [28,28])    
    #mean_img = mean_img.astype(np.float32)    
    if run_mode == "train" :                 
        print("<<<Training Mode>>>")
        # cost optimizer
        learning_rate = 0.0001 # It seems that  Adam requires an small learning rate
        with tf.device(device_name) :            
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(net['loss'])
            
        with tf.Session() as sess:
            #-------------------initialization of variable of graph
            sess.run(tf.global_variables_initializer())
            #the following is used only to make tensorboard available
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            sess.run(training_init_op)                            
            for n_iterations in range(conf.NUM_ITERATIONS):                                            
                try:
                    img, label = sess.run(next_batch)
                    img_for_train = np.array([im  for im in img])                        
                    sess.run(optimizer, feed_dict={net['x']: img_for_train, net['y_true']: label})                        
                    #each 100 iterations print loss 
                    if n_iterations % 100 == 0:
                        loss = sess.run(net['loss'], feed_dict={net['x']: img_for_train, net['y_true']: label})
                        print("Training iteration: {}, loss: {}".format(n_iterations, loss))
                    #each SNAPSHOT_TIME iterations save snapshots
                    if n_iterations % conf.SNAPSHOT_TIME == 0 or n_iterations == conf.NUM_ITERATIONS - 1 :
                        saved_path = saver.save(sess, conf.SNAPSHOT_PREFIX, global_step=n_iterations)                            
                        print("saved: {}".format(saved_path))
                    #each TEST_TIME iterations  print accuracsy
                    if n_iterations % conf.TEST_TIME == 0:                            
                        #------------- changing 
                        sess.run(testing_init_op)  
                        test_loss = 0
                        test_acc = 0                          
                        for i_test in range(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST):
                            img, label = sess.run(next_batch_test)
                            img_for_test = np.array([im for im in img])                        
                            loss = sess.run(net['loss'], feed_dict={net['x']: img_for_test, net['y_true']: label})
                            acc = sess.run(net['acc'], feed_dict={net['x']: img_for_test, net['y_true']: label})
                            test_loss = test_loss + loss
                            test_acc = test_acc + acc
                        test_loss = test_loss / float(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST)
                        test_acc = test_acc / float(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST)
                        print("Testing  loss: {} acc: {} ".format(test_loss, test_acc))                                                
                except tf.errors.OutOfRangeError:
                    sess.run(training_init_op)
                    
    elif run_mode == "test":
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            sess.run(testing_init_op)            
            print ("restoring .....\n")
            #saver = tf.train.import_meta_graph('./trained/mnist_net-3000.meta')
            #saver.restore(sess,tf.train.latest_checkpoint('./trained/'))#
            saver.restore(sess, conf.SNAPSHOT_PREFIX + '-9999')
            test_loss = 0
            test_acc = 0
            for i_test in range(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST):
                img, label = sess.run(next_batch_test)
                img_for_test = np.array([im for im in img])
                loss = sess.run(net['loss'], feed_dict={net['x']: img_for_test, net['y_true']: label})
                acc = sess.run(net['acc'], feed_dict={net['x']: img_for_test, net['y_true']: label})
                test_loss = test_loss + loss
                test_acc = test_acc + acc
            test_loss = test_loss / float(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST)
            test_acc = test_acc / float(conf.ESTIMATED_NUMBER_OF_BATCHES_TEST)
            print("Testing  loss: {} acc: {} ".format(test_loss, test_acc))
