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
r = np.random.randint(x_train.shape[0])
plt.imshow(x_train[r],cmap = "gray")

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

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten()) #Passage de tableau 2D à 1D

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# =============================================================================
# Optimizer et loss function
# =============================================================================
model.compile(loss = "categorical_crossentropy",optimizer = "adam", metrics = ['accuracy'])

# =============================================================================
# Affichage de la structure du réseau dans la console
# =============================================================================
print(model.summary())

# =============================================================================
# Apprentissage
# =============================================================================
hist = model.fit(X_train,Y_train_cat,epochs = 10,batch_size = 128)

# =============================================================================
# Sauvegarder le modèle
# =============================================================================
model.save("Model_mnist.h5")

# =============================================================================
# Affichage de l'historique de l'apprentissage
# =============================================================================
figure = plt.figure()

ax1 = plt.subplot(121)
ax1.plot(hist.history["loss"],label = "Training Loss")
plt.xlabel("Nombre d'itérations")
plt.ylabel("Loss function")
plt.legend()

ax2 = plt.subplot(122)
ax2.plot(hist.history["acc"],label = "Training Accuracy")
plt.xlabel("Nombre d'itérations")
plt.ylabel("Accuracy")
plt.legend()

# =============================================================================
# Accuracy sur la base de test
# =============================================================================

score = model.evaluate(X_test, Y_test_cat,batch_size = 100)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# =============================================================================
# Accuracy sur la base de test
# =============================================================================

prediction = model.predict(X_test)
prediction = np.argmax(prediction,axis = 1)
print("Accuracy on test set: " + str(np.round(np.sum(prediction == y_test)/x_test.shape[0]*100,3)) + " %")
#===================
