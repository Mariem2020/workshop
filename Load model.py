# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:02:47 2020

@author: HP
"""
from keras.datasets import mnist
from keras.utils import to_categorical
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
#=============================================================================
# load model
#==============================================================================
model = load_model('model_mnist_cnn.h5')
# summarize model.
model.summary()

# ============================================================================
# load dataset
#==============================================================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255
Y_train_cat = to_categorical(y_train,num_classes = 10)
Y_test_cat = to_categorical(y_test,num_classes = 10)

# ============================================================================
# evaluate the model
# ============================================================================
score = model.evaluate(X_test, Y_test_cat, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
