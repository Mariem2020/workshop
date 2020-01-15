# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:47:21 2020

@author: HP
"""

# =============================================================================
# Importation des Bibliothèques 
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

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
# Affichage de la structure du réseau dans la console
# =============================================================================

print(model.summary())



