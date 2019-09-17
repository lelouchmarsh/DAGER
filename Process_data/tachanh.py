import pandas as pd
import shutil
from shutil import copyfile
import os
malepath="D:/Gender Data/male1/"
femalepath="D:/Gender Data/female1/"
data=pd.read_csv('D:\\data\\meta.csv')
i=len(data)
counter=0
for a in range(0,i+1):
	if data['gender'][a]=='female':
		shutil.copy2("D:/data/"+data['path'][a],femalepath)
		print (str(counter) + "process")
		counter=counter+1
	if data['gender'][a]=='male':
		shutil.copy2("D:/data/"+data['path'][a],malepath)
		print (str(counter) + "process")
		counter=counter+1