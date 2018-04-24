#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: A list of function to create tfrecords
"""

from skimage import io, transform
import os
import struct
import sys
import numpy as np
import tensorflow as tf
import random

# %%
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# %%
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# %%
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#load mnist from the binay files
def loadMNIST(pathname = ".",  dataset = "train", shuffle = True):
    if dataset == "train":
        fname_images = os.path.join(pathname, 'train-images.idx3-ubyte')
        fname_labels = os.path.join(pathname, 'train-labels.idx1-ubyte')
    elif dataset == "test" :
        fname_images = os.path.join(pathname, 't10k-images.idx3-ubyte')
        fname_labels = os.path.join(pathname, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("use loadMNIST with train | test")
        
    with open(fname_labels, 'rb') as f_lbl:
        magic, num  = struct.unpack(">II", f_lbl.read(8))
        labels = np.fromfile(f_lbl, dtype = np.uint8)
    
    with open(fname_images, 'rb') as f_img:
        magic, num, rows, cols = struct.unpack(">IIII", f_img.read(16))
        images = np.fromfile(f_img, dtype = np.uint8).reshape(num, rows, cols)
    
    if shuffle:
        inds = list(range(len(labels)))
        np.random.shuffle(inds)
        images = images[inds]        
        labels = labels[inds]    
    return images, labels

#creating tfrecords
def createTFRecord(images, labels, tfr_filename):
    h = images.shape[1]
    w = images.shape[2]    
    writer = tf.python_io.TFRecordWriter(tfr_filename)    
    assert len(images) == len(labels)
    mean_image = np.zeros([h,w], dtype=np.float32)
    for i in range(len(images)):        
        print("---{}".format(i))                
        #print("{}label: {}".format(label[i]))
        #create a feature
        feature = {'train/label': _int64_feature(labels[i]), 
                   'train/image': _bytes_feature(tf.compat.as_bytes(images[i, :, :].tostring()))}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + images[i, :, :]

    mean_image = mean_image / len(images)        
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image
# %%
#create TFRecords for MNIST dataset
def createMnistTFRecord(str_path):    
    #------------- creating train data
    images, labels = loadMNIST(str_path, dataset="train", shuffle = True)    
    tfr_filename = os.path.join(str_path, "train.tfrecords")
    training_mean = createTFRecord(images, labels, tfr_filename)
    print("train_record saved at {}.".format(tfr_filename))
    #-------------- creating test data    
    images, labels = loadMNIST(str_path, dataset="test", shuffle = True)  
    tfr_filename = os.path.join(str_path, "test.tfrecords")
    createTFRecord(images, labels, tfr_filename)
    print("test_record saved at {}.".format(tfr_filename))
    #saving training mean
    mean_file = os.path.join(str_path, "mean.dat")
    print("mean_file {}".format(training_mean.shape))
    training_mean.tofile(mean_file)
    print("mean_file saved at {}.".format(mean_file))  

#---------parser_tfrecord for mnist
def parser_tfrecord(serialized_example):
    features = tf.parse_example([serialized_example],
                                features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64),
                                        })
    image = tf.decode_raw(features['train/image'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    image = tf.cast(image, tf.float32)
    image = image * 1.0 / 255.0    
    
    label = tf.one_hot(tf.cast(features['train/label'], tf.int32), 10)
    label = tf.reshape(label, [10])
    label = tf.cast(label, tf.float32)
    return image, label
