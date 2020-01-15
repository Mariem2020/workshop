# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:50:51 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:21:07 2019

@author: HP
"""

# =============================================================================
# Importation des Bibliothèques 
# =============================================================================
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Importation de la base de données MNIST (ce sont des images de chiffres, hello world du deep learning)
# =============================================================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =============================================================================
# Affichage des chiffres
# =============================================================================

#r = np.random.randint(x_train.shape[0])

#plt.imshow(x_train[r],cmap = "gray")

# =============================================================================
# Reconfiguration des structures des données #reshape data to fit model
# =============================================================================

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# =============================================================================
# Conversion en float32 & Normalisation des données
# =============================================================================
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')

X_train = X_train/255
X_test = X_test/255

# =============================================================================
# Passage des données (0,1,2,3...) en catégories. Exemple : 3 devient (0,0,0,1,0,0,0,0,0,0)
# =============================================================================

Y_train_cat = to_categorical(y_train,num_classes = 10)

Y_test_cat = to_categorical(y_test,num_classes = 10)

# =============================================================================
# Construction du réseau
# =============================================================================
activation='sigmoid'
model = Sequential()

# - Full connection
model.add(Flatten(input_shape = (28,28,1))) #  2D to 1D
model.add(Dense(200,activation = activation, kernel_initializer = "RandomUniform")) # hidden layer1 keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
model.add(Dense(100,activation = activation, kernel_initializer = "RandomUniform")) # hidden layer2
model.add(Dense(60,activation = activation, kernel_initializer = "RandomUniform")) # hidden layer3
model.add(Dense(30,activation = activation, kernel_initializer = "RandomUniform")) # hidden layer4
model.add(Dense(10,activation = "softmax", kernel_initializer = "RandomUniform"))   # ouput layer5
print(model.summary())

# =============================================================================
# Affichage de la structure du réseau dans la console
# =============================================================================
print(model.summary())

# =============================================================================
# Optimizer et loss function
# =============================================================================
opt = optimizers.Adam(lr = 0.003)
model.compile(loss = "categorical_crossentropy",optimizer = opt, metrics = ['accuracy'])

# =============================================================================
# Apprentissage
# =============================================================================
batch_size = 100
epochs = 500
# Apprentissage du modèle
hist = model.fit(X_train,Y_train_cat,epochs = epochs,batch_size = batch_size,validation_data=(X_test, Y_test_cat), verbose=1)
# =============================================================================
# Sauvegarder le modèle
# =============================================================================
model.save("/content/drive/My Drive/code workshop/model_mnist_5layer_Relulrdecay.h5")
model.save_weights("/content/drive/My Drive/code workshop/weightmodel_mnist_5layer_Relulrdecay.h5")
print("Saved model and weights")
# =============================================================================
# Accuracy sur la base de test
# =============================================================================
score = model.evaluate(X_test, Y_test_cat,batch_size = 100)#, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# =============================================================================
# Accuracy sur la base de test
# =============================================================================

prediction = model.predict(X_test)
prediction = np.argmax(prediction,axis = 1)

print("Accuracy on test set: " + str(np.round(np.sum(prediction == y_test)/x_test.shape[0]*100,3)) + " %")
#===================
# =============================================================================
# Affichage de l'historique de l'apprentissage
# =============================================================================
#Graphiques  et  résultats
history_dict = hist.history
history_dict.keys()
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc'] #ce qui nous intéresse
epochs = range(1, len(acc) + 1) 
#perte
plt.plot(epochs, loss_values, 'r-', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("/content/drive/My Drive/code workshop/ANN-Loss.png", bbox_inches="tight", dpi=600)
plt.show()
#précision
plt.clf()
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy FNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("/content/drive/My Drive/code workshop/ANN-Accuracy.png", bbox_inches="tight", dpi=600)
plt.show()  #!!!!   affiche et remet à zéro => sauvegarder avant 
