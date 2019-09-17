import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
def chuanhoa(a:str ):
    index=a.find('-',1)
    if index==1:
        a='0'+a
    if index>2:
        for i in range(0,index-2):
            a=a.replace(a[0],'',1)
    # if a[len(a)-1]=='0':
    #   a=a.replace(a[len(a)-1],'1',1)
    return a
cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

imdb_mat = 'D:\\data\\imdb_crop\\imdb_crop\\imdb.mat'
wiki_mat = 'D:\\data\\wiki_crop\\wiki_crop\\wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb_crop/' + path[0])

for path in wiki_full_path:
    wiki_path.append('wiki_crop/' + path[0])

imdb_genders = []
wiki_genders = []

# for n in range(len(imdb_gender)):
#     if imdb_gender[n] == 1:
#         imdb_genders.append('male')
#     else:
#         imdb_genders.append('female')

# for n in range(len(wiki_gender)):
#     if wiki_gender[n] == 1:
#         wiki_genders.append('male')
#     else:
#         wiki_genders.append('female')

imdb_dob = []
wiki_dob = []

for file in imdb_path:
    temp = file.split('_')[3]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]

    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    
    imdb_dob.append('-'.join(temp))

for file in wiki_path:
    wiki_dob.append(file.split('_')[2])


imdb_age = []
wiki_age = []

for i in range(len(imdb_dob)):
    try:
        imdb_dob[i]=chuanhoa(imdb_dob[i])
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        print(d1)
        print(d2)
        rdelta = relativedelta(d2, d1) #Time distance
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)
    print(d1)
    print(d1.year)
    print(imdb_dob[i])
    print(type(imdb_dob[i]))
    
    break

# for i in range(len(wiki_dob)):
#     try:
#         d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
#         d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
#         rdelta = relativedelta(d2, d1)
#         diff = rdelta.years
#     except Exception    as ex:
#         print(ex)
#         diff = -1
#     wiki_age.append(diff)