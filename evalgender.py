import pandas as pd
import numpy as np
import cv2
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
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


model = get_model(n_classes=2)
# model = model.load_model("F:/Project_Gender/gender_weight1.h5")
model.load_weights("F:/Project_Gender/gender_weight1.h5")


DIR="F:/Project_Gender/test/"
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
      rescale=1/255)  #included in our dependencies

print(os.listdir(DIR))
test_generator=test_datagen.flow_from_directory( DIR, #test path
                                                 target_size=(200,200),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=False)


filenames = test_generator.filenames
nb_samples = len(filenames)
# print(nb_samples)
# print(test_generator.classes[1]) # Index của bức ảnh
predict = model.predict_generator(test_generator,nb_samples)

print (type(test_generator))
# plt.imshow(test_generator[0])
# plt.show()
print(predict)

print(nb_samples )

k=0
for i in range(0,nb_samples):
  if (predict[i][test_generator.classes[i]]==np.amax(predict[i])): #top v1 - accuracy
    k=k+1
    print (i)
    print(test_generator.classes[i])
image = cv2.imread ("F:/Project_Gender/test/female/1.jpg")
image = image/255
image = np.expand_dims(image, axis=0)
predict = model.predict (image)
print (predict)
print (type(predict))
print (len(predict))
print (predict[0][0])




# q=0
# for i in range(0,nb_samples+1):
#   if (predict[i][test_generator.classes[i]]> np.unique(predict[i])[-3]): #top v3 - accuracy
#     q=q+1
# print(q)
# print(k/nb_samples)   #Dùng cho AGE
