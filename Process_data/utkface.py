import os
import numpy as np
import shutil
from shutil import copyfile
DIR1="D:\\Gender Data\\male"
DIR2="D:\\Gender Data\\female"
DIR="D:\\data\\UTKFace\\face"
images = os.listdir(DIR)
for image in images:
	path = os.path.join(DIR,image)
	print(path)
	index=(image[image.find('_',1)+1]) #Detect Gender
	print(index)
	print(type(index))
	if index=='0':
		print("male")
		# shutil.copy2(path,DIR1)
	if index=='1':
		print("female")
		# shutil.copy2(path,DIR2)
	break