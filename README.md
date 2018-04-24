# cnn_mnist
# Steps #
## 1. Download mnist data  ##
mnist data can be downloaded from:
* http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
* http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
* http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
* http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

Decompress the files into <mnist_path>. Set then the variable DATA_DIR, in configuration.py, to <mnist_path>  

## 2. Create tfrecords ## 
python3.6 create_mnist_data 
## 3. Train model ## 
python3.6 train_mnist_net.py train <gpu | cpu>
## 4. Test model ## 
python3.6 train_mnist_net.py test <gpu | cpu>


