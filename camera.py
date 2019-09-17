import numpy
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

gender_list = os.listdir ("F:/Project_Gender/test/")
age_list = os.listdir("F:/Project_Gender/age_test/")
def get_model(n_classes):

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


modelgender = get_model(n_classes=2)
modelgender.load_weights("F:/Project_Gender/gender_weight1.h5")
modelage = get_model(n_classes = 8)
modelage.load_weights("F:/Project_Gender/age_weight1.h5")

####### Lấy model và weights #############

import face_recognition
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
WIDTH=200
HEIGHT=200

def label (crop_img):
	crop_img = crop_img/255
	crop_img = np.expand_dims(crop_img, axis=0)
	predict = modelgender.predict (crop_img)
	if predict[0][0] > predict[0][1] :
		label = "female"
	if predict[0][1] > predict[0][0] :
		label = "male"
	return (label)
####### Xử lý ảnh và trả về label #######

def labelage (crop_img):
	crop_img = crop_img/255
	crop_img = np.expand_dims(crop_img, axis=0)
	predict = modelage.predict (crop_img)
	print (predict)
	for i in range (0,8):
		if predict[0][i] == np.amax ( predict [0]):
			labelage = age_list[i]
			break
	return (labelage)
####### label age #####################


def draw_label(image, point, pointk, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.3, thickness=5):
    size = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(image, point, pointk, (255, 0, 0),thickness)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

### Vẽ hình #################

def labeled  (image):
	global label
	# image = face_recognition.load_image_file(your_file)
	face_locations = face_recognition.face_locations(image)

	for i in range(0,len( face_locations )):
		x = face_locations[i][0] #x = 98
		xk = face_locations[i][2] #x+ = 253
		y = face_locations[i][3] # y = 356
		yk = face_locations[i][1] #y+ =  511 

		point = y,x
		pointk = yk,xk	

		crop_img  = image [ x:xk , y:yk ]
		crop_img  = cv2.resize(crop_img, (WIDTH,HEIGHT) )
		# plt.imshow(crop_img)
		# plt.show()
		labels  =  label (crop_img) + "  " + labelage (crop_img)
		print (labels)

		draw_label(image = image, point = point, pointk =	pointk, label = labels)
	return (image)
	# plt.imshow(image)
	# plt.show()

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    labeled (frame)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()