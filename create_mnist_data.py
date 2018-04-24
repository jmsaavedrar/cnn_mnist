"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: A list of function to create tfrecords
mnist data can be downloaded from:
     - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
     - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
     - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
     - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

Decompress each file into <mnist_path>. Then set the variable DATA_DIR = <mnist_path>  

"""

import utils.data as udata
import matplotlib.pyplot as plt
import  numpy as np
import configuration as conf

if __name__ == '__main__':        
    udata.createMnistTFRecord(conf.DATA_DIR)
    print("tfrecords created for mnist")
