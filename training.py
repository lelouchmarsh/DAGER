import os
import pandas as pd
import os
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K
def get_model(n_classes=1):

    base_model = ResNet50(weights='imagenet', include_top=False)

    for layer in base_model.layers:
       layer.trainable = True

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    if n_classes == 1:
        x = Dense(n_classes, activation="sigmoid")(x)
    else:
        x = Dense(n_classes, activation="softmax")(x)

    base_model = Model(base_model.input, x, name="base_model")
    if n_classes == 1:
        base_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer="adam")
    else:
        # base_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")
        base_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return base_model
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      vertical_flip=True,
      rescale=1/255) #included in our dependencies

train_generator=train_datagen.flow_from_directory(r'/data/facereg/face_aligned/gender_data/train', #Train path
                                                 target_size=(200,200),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 )               
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      vertical_flip=True,
      rescale=1/255)

validation_generator= validation_datagen.flow_from_directory(r'/data/facereg/face_aligned/gender_data/val', #val path
                                                 target_size=(200,200),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False,
                                                 )
model = get_model(n_classes=2) 
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n // train_generator.batch_size



filepath = "/home/nguyenquan/Project_Gender/gender_weight1.h5"

model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
early = EarlyStopping(monitor="acc", mode="max", patience=25) #25 epoch không improve val_acc thì dừng
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=5) #5 epoch
callbacks_list = [checkpoint,early,reduce_on_plateau]



history=model.fit_generator(generator = train_generator, epochs = 50, callbacks = callbacks_list,steps_per_epoch = step_size_train)

history.history['acc']


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show

