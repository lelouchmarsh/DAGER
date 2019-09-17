# Importing dependencies
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import shutil

# Loading dataset
meta = pd.read_csv('D:\\data\\meta.csv')

# Dropping gender column
meta = meta.drop(['gender'], axis=1)

# Filtaring dataset
meta[meta['age'] < 0] = 0

# Making the directory structure
for i in range(102):
    output_dir_train = 'D:/dataset/age/' + str(i)
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)

# Finally making the training and testing set
counter = 0

for i in range(0,len(meta)+1):
    if meta['age'][i]>0:
        shutil.copy2("D:/data/"+meta['path'][i], 'D:/dataset/age/'+str(meta['age'][i]))
        counter= counter+1
        print(str(counter)+'processcing')
